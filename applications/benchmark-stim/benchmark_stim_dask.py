"""Cluster-parallel version of ``benchmark_stim.py`` (Dask + SLURM).

Same honest-failure / feasibility-region pipeline and **identical CSV columns** as
``benchmark_stim.py`` (``p_ent,width,depth,p_failed_round,p_false_reject``), so
``applications/plot_veriphix_heatmaps.py`` plots the output unchanged.

The key observation that makes this embarrassingly parallel: the circuits are **not**
files on disk — each ``(width, depth)`` tile regenerates its own fixed Clifford circuit
deterministically from ``base_seed`` via ``PCG64(base_seed).jumped(width*1009+depth)``.
So one Dask task = one ``(width, depth, p_ent)`` cell, carrying only a handful of ints
and floats; the worker rebuilds the pattern from the seed. No pre-generation, no circuit
shipping. This mirrors the gospel benchmark (``applications/gospel/benchmark/simulation.py``)
and the old ACES driver, which fan out one Dask future per circuit/order.

Each cell is submitted as an independent future; results are written to CSV as futures
complete (not in submission order), and an existing CSV is used to **resume** — already
computed ``(p_ent, width, depth)`` cells are skipped.

Usage — local (LocalCluster, uses all CPU cores)
-------------------------------------------------
    python applications/benchmark-stim/benchmark_stim_dask.py --widths 2,3,4 --depths 2,4

Usage — SLURM cluster
---------------------
    python applications/benchmark-stim/benchmark_stim_dask.py \\
        --widths 8,9,10 --depths 16 \\
        --walltime 2 --memory 8 --cores 4 --port 8787 --scale 20
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import sys
import time
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Annotated

import dask.distributed
import numpy as np
import stim
import typer
from dask_jobqueue import SLURMCluster
from graphix import Pattern, command
from graphix.command import CommandKind
from graphix.sim.statevec import Statevec
from numpy.random import PCG64, Generator

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12
from veriphix.sampling_circuits.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from veriphix.uncorrelated_depolarising_noise_model import UncorrelatedDepolarisingNoiseModel
from veriphix.verifying import generate_eigenstate

# The Stim transpiler is vendored in the ACES experiment folder (not a pip-installed
# package). Workers don't run this sys.path hack, so the module must be shipped to them
# explicitly (see upload_file in main); _ACES_DIR / _STIM_MODULE are captured here for that.
_ACES_DIR = Path(__file__).resolve().parent.parent / "aces"
_STIM_MODULE = _ACES_DIR / "stim_pauli_preprocessing.py"
sys.path.insert(0, str(_ACES_DIR))
from stim_pauli_preprocessing import BasicState, pattern_to_stim_circuit

app = typer.Typer(add_completion=False)

# Dask logs a CommClosedError from the heartbeat coroutine during teardown; it is benign
# (the run is already finished) but alarming. Silence the comm layer's error logger.
logging.getLogger("distributed.comm").setLevel(logging.CRITICAL)
logging.getLogger("distributed.client").setLevel(logging.CRITICAL)

# CSV columns are frozen to match benchmark_stim.py / the heatmap plotter.
CSV_FIELDS = ["p_ent", "width", "depth", "p_failed_round", "p_false_reject"]


# ── helpers (copied from benchmark_stim.py so the worker is self-contained) ──────


def state_to_basic_state(state: object) -> BasicState:
    bs = BasicState.try_from_statevector(Statevec(state).psi)
    if bs is None:
        raise ValueError(f"Not a basic state: {state}")
    return bs


def x_basis_measurement_pattern(clean_pattern: Pattern) -> Pattern:
    """Re-attach an X-basis ``M`` to every bare ``BaseM`` of a flow-removed pattern."""
    pattern = Pattern(input_nodes=clean_pattern.input_nodes)
    for cmd in clean_pattern:
        if cmd.kind == CommandKind.M:
            pattern.add(command.M(node=cmd.node))
        else:
            pattern.add(cmd)
    return pattern


def _round_fail_pool(run: object, stim_pattern: Pattern, noise_model: object, n_total: int) -> np.ndarray:
    """Batch-sample ``n_total`` independent test rounds of one test run.

    Returns a boolean array ``(n_total,)``: ``True`` where the round fails (some trap
    parity, XORed with the stabiliser sign, is 1).
    """
    input_state: dict[int, BasicState] = {}
    fixed_states: dict[int, BasicState] = {}
    for node, state in run.input_state.items():
        bs = state_to_basic_state(state)
        if node in stim_pattern.input_nodes:
            input_state[node] = bs
        else:
            fixed_states[node] = bs
    circuit, measure_indices = pattern_to_stim_circuit(
        stim_pattern, input_state=input_state, noise_model=noise_model, fixed_states=fixed_states
    )
    samples = np.asarray(circuit.compile_sampler().sample(shots=n_total))
    sign_flip = int(run.stabilizer.sign == -1)
    round_fail = np.zeros(n_total, dtype=bool)
    for trap in run.traps:
        cols = [measure_indices[node] for node in trap]
        parity = (samples[:, cols].sum(axis=1) & 1) ^ sign_flip
        round_fail |= parity.astype(bool)
    return round_fail


# ── fast FK12 test runs (Part A — see TODO.md) ──────────────────────────────────


@dataclass(frozen=True)
class _FastRun:
    """Drop-in for ``veriphix.verifying.TestRun`` exposing only what ``_round_fail_pool``
    reads: ``input_state``, ``stabilizer`` (for its ``.sign``), and ``traps``."""

    input_state: dict[int, object]
    stabilizer: stim.PauliString
    traps: frozenset[frozenset[int]]


def fk12_fast_runs(graph: object, coloring: list[set[int]]) -> list[_FastRun]:
    """Closed-form FK12 X-basis test runs in O(|V|+|E|), bypassing ``build_stabilizer``.

    For each colour of the bipartite (independent-set) coloring, the merged trap
    stabilizer is, in closed form:

        X on every node of this colour,  Z on every opposite-colour node (deg>=1),  sign +1

    (``merge`` overlays supports rather than multiplying Paulis, so opposite-colour nodes
    get ``Z`` with no degree-parity cancellation.) This reproduces the exact
    ``stim.PauliString`` that ``build_stabilizer`` returns — verified against it — so the
    existing ``generate_eigenstate`` yields an identical ``input_state``. Replaces the
    O(|V|^~3) tableau/conjugate/merge machinery with degrees + colour lookup.
    """
    nodes = list(graph.nodes)
    deg = dict(graph.degree())
    runs: list[_FastRun] = []
    for colour in coloring:
        letters = [
            "X" if n in colour else ("Z" if deg[n] >= 1 else "I") for n in nodes
        ]
        stab = stim.PauliString("".join(letters))  # sign +1
        input_state = dict(zip(nodes, generate_eigenstate(stab)))
        traps = frozenset(frozenset([n]) for n in colour)
        runs.append(_FastRun(input_state=input_state, stabilizer=stab, traps=traps))
    return runs


# ── result types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellResult:
    p_ent: float
    width: int
    depth: int
    p_failed_round: float
    p_false_reject: float
    nodes: int = 0
    edges: int = 0
    build_s: float = 0.0
    sample_s: float = 0.0
    elapsed_s: float = 0.0  # wall-time for this cell; not written to CSV


@dataclass(frozen=True)
class CellFailure:
    p_ent: float
    width: int
    depth: int
    error: str
    elapsed_s: float = 0.0


# ── work unit ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """One ``(width, depth, p_ent)`` tile — the unit of Dask parallelism.

    Carries only serialisable scalars; the worker regenerates the fixed Clifford
    circuit from ``base_seed`` (identical to ``benchmark_stim.simulate_cell``).
    """

    width: int
    depth: int
    p_ent: float
    n_shots: int
    test_rounds: int
    threshold: int
    base_seed: int

    def execute(self) -> CellResult | CellFailure:
        t0 = time.monotonic()
        try:
            t_b0 = time.monotonic()
            # Deterministic per (width, depth): same fixed circuit across noise levels.
            rng = Generator(PCG64(self.base_seed).jumped(self.width * 1009 + self.depth))
            pattern = generate_random_pauli_pattern(
                nqubits=self.width, nlayers=self.depth, order=ConstructionOrder.Canonical, rng=rng
            )
            coloring = list(get_bipartite_coloring(pattern))
            # autogen=False + manual prep skips create_trappified_scheme, whose
            # build_stabilizer call is the ~O(|V|^3) bottleneck. The cheap pattern prep
            # (preprocess + blind, <10ms) is reused; test runs come from the closed form.
            client = Client(
                pattern=pattern,
                secrets=Secrets(r=False, a=False, theta=False),
                protocol=FK12(manual_colouring=coloring),
                rng=rng,
                autogen=False,
            )
            client.preprocess_pattern()
            client.create_blind_patterns(secrets=Secrets(r=False, a=False, theta=False), rng=rng)
            stim_pattern = x_basis_measurement_pattern(client.clean_pattern)
            test_runs = fk12_fast_runs(client.graph, coloring)
            build_s = time.monotonic() - t_b0

            t_s0 = time.monotonic()
            n_total = self.n_shots * self.test_rounds
            noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=self.p_ent)
            pools = [_round_fail_pool(run, stim_pattern, noise_model, n_total) for run in test_runs]

            # Each round independently picks a test run (FK12.sample_test_run is uniform).
            choice = rng.integers(0, len(pools), size=n_total)
            fails = np.empty(n_total, dtype=bool)
            for k, pool in enumerate(pools):
                mask = choice == k
                fails[mask] = pool[: int(mask.sum())]
            fails = fails.reshape(self.n_shots, self.test_rounds)
            sample_s = time.monotonic() - t_s0

            nr_failed = fails.sum(axis=1)
            return CellResult(
                p_ent=self.p_ent,
                width=self.width,
                depth=self.depth,
                p_failed_round=float(fails.mean()),
                p_false_reject=float((nr_failed > self.threshold).mean()),
                nodes=int(pattern.n_node),
                edges=int(sum(1 for c in pattern if c.kind == CommandKind.E)),
                build_s=build_s,
                sample_s=sample_s,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return CellFailure(
                p_ent=self.p_ent,
                width=self.width,
                depth=self.depth,
                error=str(exc),
                elapsed_s=time.monotonic() - t0,
            )


# ── cluster helpers (mirror applications/gospel/benchmark/simulation.py) ──────────


def _get_cluster(
    walltime: int | None,
    memory: int | None,
    cores: int | None,
    port: int | None,
    scale: int | None,
) -> dask.distributed.deploy.cluster.Cluster:
    if walltime is None and memory is None and cores is None:
        cluster: dask.distributed.deploy.cluster.Cluster = dask.distributed.LocalCluster()
    else:
        for name, val in [("--walltime", walltime), ("--memory", memory), ("--cores", cores), ("--port", port), ("--scale", scale)]:
            if val is None:
                raise ValueError(f"{name} is required for SLURM")
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=cores,
            memory=f"{memory}GB",
            walltime=f"{walltime}:00:00",
            scheduler_options={"dashboard_address": f":{port}"},
            # Put the vendored ACES module dir on the workers' PYTHONPATH so every SLURM
            # job (including ones that join later via scale) can import
            # stim_pauli_preprocessing natively. Backs up the upload_file in main.
            job_script_prologue=[f"export PYTHONPATH={_ACES_DIR}:$PYTHONPATH"],
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    """Return the set of (p_ent, width, depth) already present in the CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {(row["p_ent"], row["width"], row["depth"]) for row in csv.DictReader(f)}


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.2f}s"


# ── main ──────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    widths:      Annotated[str, typer.Option(help="Comma-separated widths")] = "8,9,10",
    depths:      Annotated[str, typer.Option(help="Comma-separated depths")] = "16",
    ent_errors:  Annotated[str, typer.Option(help="Comma-separated entanglement error probs")] = "1e-3",
    shots:       Annotated[int, typer.Option()] = 100,
    test_rounds: Annotated[int, typer.Option()] = 100,
    threshold:   Annotated[int, typer.Option()] = 0,
    out_csv:     Annotated[Path, typer.Option()] = Path("applications/benchmark-stim/benchmark_stim_results.csv"),
    seed:        Annotated[int, typer.Option()] = 42,
    walltime:    Annotated[int | None, typer.Option(help="SLURM: walltime in hours")] = None,
    memory:      Annotated[int | None, typer.Option(help="SLURM: memory in GB")] = None,
    cores:       Annotated[int | None, typer.Option(help="SLURM: cores per job")] = None,
    port:        Annotated[int | None, typer.Option(help="SLURM: dashboard port")] = None,
    scale:       Annotated[int | None, typer.Option(help="Number of workers")] = None,
    smoke:       Annotated[bool, typer.Option()] = False,
) -> None:
    """Sweep (width, depth) x p_ent across a Dask cluster; write the honest-failure CSV."""
    if smoke:
        widths, depths, ent_errors, shots, test_rounds = "2,3", "2,3", "1e-2", 20, 20

    width_list = _parse_ints(widths)
    depth_list = _parse_ints(depths)
    ent_list = _parse_floats(ent_errors)

    # One Cell per (width, depth, p_ent). dims ordered (w fastest within a depth) to
    # match benchmark_stim.py.
    dims = [(w, d) for d in depth_list for w in width_list]
    cells = [
        Cell(
            width=w,
            depth=d,
            p_ent=p,
            n_shots=shots,
            test_rounds=test_rounds,
            threshold=threshold,
            base_seed=seed,
        )
        for (w, d) in dims
        for p in ent_list
    ]
    n_cells_total = len(cells)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_csv)
    if done:
        typer.echo(f"Resuming: {len(done)} cells already in {out_csv}")
    cells = [c for c in cells if (str(c.p_ent), str(c.width), str(c.depth)) not in done]

    typer.echo(
        f"grid: {len(width_list)} widths x {len(depth_list)} depths x {len(ent_list)} noise levels "
        f"= {n_cells_total} cells ({len(cells)} to run); "
        f"shots={shots}, test_rounds={test_rounds} ({shots * test_rounds} samples/colour)"
    )
    if not cells:
        typer.echo("Nothing to do — all cells already present.")
        return

    is_new = not out_csv.exists() or out_csv.stat().st_size == 0

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)
    typer.echo(f"Dask dashboard: {dask_client.dashboard_link}")

    # Ship the vendored Stim transpiler to every worker (current and future). It is not a
    # pip-installed package, so workers cannot import `stim_pauli_preprocessing` on their
    # own — without this, Cell.execute fails to deserialise on the worker and the run
    # silently stalls (this is the one thing the gospel benchmark doesn't need).
    if _STIM_MODULE.exists():
        dask_client.upload_file(str(_STIM_MODULE))
        typer.echo(f"Uploaded {_STIM_MODULE.name} to workers.")

    # Per-dimension timing accumulators (aggregated from CellResult as futures land).
    dim_nodes: dict[tuple[int, int], int] = {}
    dim_edges: dict[tuple[int, int], int] = {}
    dim_build: dict[tuple[int, int], float] = {}
    dim_sample: dict[tuple[int, int], float] = {}
    dim_count: dict[tuple[int, int], int] = {}

    n_ok = n_fail = 0
    loop_start = time.monotonic()

    try:
        futures = [dask_client.submit(Cell.execute, c, pure=False) for c in cells]

        with out_csv.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
            if is_new:
                writer.writeheader()

            for fut in dask.distributed.as_completed(futures):
                try:
                    report = fut.result()
                except Exception as exc:
                    typer.echo(f"Future error: {exc}")
                    n_fail += 1
                    continue
                finally:
                    fut.release()  # free the worker-side result promptly

                n_done = n_ok + n_fail + 1
                wall_elapsed = time.monotonic() - loop_start
                throughput = n_done / wall_elapsed  # cells/s (reflects parallelism)
                remaining = len(cells) - n_done
                eta_str = _fmt(remaining / throughput) if throughput > 0 else "?"

                if isinstance(report, CellResult):
                    writer.writerow(
                        {
                            "p_ent": report.p_ent,
                            "width": report.width,
                            "depth": report.depth,
                            "p_failed_round": report.p_failed_round,
                            "p_false_reject": report.p_false_reject,
                        }
                    )
                    csvfile.flush()
                    n_ok += 1

                    key = (report.width, report.depth)
                    dim_nodes[key] = report.nodes
                    dim_edges[key] = report.edges
                    dim_build[key] = dim_build.get(key, 0.0) + report.build_s
                    dim_sample[key] = dim_sample.get(key, 0.0) + report.sample_s
                    dim_count[key] = dim_count.get(key, 0) + 1

                    typer.echo(
                        f"  [{n_ok + n_fail}/{len(cells)}] w={report.width:>2} d={report.depth:>2} "
                        f"p={report.p_ent:.1e} |V|={report.nodes:>4} |E|={report.edges:>4}  "
                        f"build={report.build_s:.2f}s sample={report.sample_s:.2f}s "
                        f"cell={_fmt(report.elapsed_s)}  "
                        f"p_fail_round={report.p_failed_round:.4f} "
                        f"p_false_reject={report.p_false_reject:.3f}  ETA {eta_str}"
                    )
                elif isinstance(report, CellFailure):
                    n_fail += 1
                    typer.echo(
                        f"  ✗ [{n_ok + n_fail}/{len(cells)}] w={report.width:>2} d={report.depth:>2} "
                        f"p={report.p_ent:.1e}  t={report.elapsed_s:.1f}s  ETA {eta_str}: {report.error}"
                    )
    finally:
        # Graceful teardown — closing the client/cluster before the process exits avoids
        # the "CommClosedError ... heartbeat_worker ... Stream is closed" noise that
        # appears when an in-flight heartbeat races a hard interpreter shutdown.
        dask_client.close()
        cluster.close()

    # ── per-dimension timing summary ───────────────────────────────────────────────
    if dim_count:
        typer.echo("\n" + "=" * 96)
        typer.echo("PER-DIMENSION TIMING  (averaged over the noise levels that completed)")
        typer.echo("=" * 96)
        typer.echo(
            f"{'width':>5} {'depth':>5} {'|V|':>6} {'|E|':>6} "
            f"{'build/cell':>11} {'sample/cell':>12} {'per_cell':>10}"
        )
        for (width, depth), n in sorted(dim_count.items(), key=lambda kv: dim_nodes[kv[0]]):
            key = (width, depth)
            b = dim_build[key] / n
            s = dim_sample[key] / n
            typer.echo(
                f"{width:>5} {depth:>5} {dim_nodes[key]:>6} {dim_edges[key]:>6} "
                f"{b:>10.2f}s {s:>11.2f}s {b + s:>9.2f}s"
            )

    typer.echo(
        f"\nDone. {n_ok} results, {n_fail} failures  "
        f"(wall {_fmt(time.monotonic() - loop_start)})  ->  {out_csv}"
    )


if __name__ == "__main__":
    freeze_support()
    app()
