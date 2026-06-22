"""Stim/Clifford honest-failure benchmark for the **Clifford + Magic-State-Injection**
verification protocol (magic-blindness), Dask + SLURM parallel.

This is the circuit-model analogue of ``applications/benchmark-stim/benchmark_stim_dask.py``.
Where that script benchmarks MBQC patterns parametrised by ``(width, depth)``, this one
benchmarks Clifford+MSI computations parametrised by ``(n, t)`` (n data qubits, t
state-injection layers), following arXiv-2601.07111v3 ("Composable Verification in the
Circuit-Model via Magic-Blindness").

Why it is pure Stim (no graphix, no |T>, no blindness, no mid-circuit M)
-----------------------------------------------------------------------
The honest-failure landscape is built **only from test rounds**, and in the
magic-blindness construction test rounds are *magic-free* -> the whole computation is the
single Clifford ``G = C_{t+1} . F_t . C_t . ... . F_1 . C_1`` on ``n+t`` qubits. So:

* **No magic states.** Test rounds inject stabiliser states only; ``|T>`` never appears.
* **No blindness.** The one-time pad is a Pauli twirl; for an honest (depolarising) server
  ``p_failed_round`` is pad-invariant, so we drop it (exactly as ``benchmark_stim.py`` runs
  with ``Secrets(r=False, a=False, theta=False)``).
* **No mid-circuit measurement.** Magic-free injection disentangles each ancilla, and the
  noise is gate-based, so deferring every measurement to the end is distribution-identical.
  We therefore treat ``G`` as one Clifford unitary on ``n+t`` qubits and measure all qubits
  at the end.

Per test round (RandomTraps over ``[n+t]``):
  * sample a random non-empty subset ``S`` of the ``n+t`` qubits;
  * its trap stabiliser is ``S_hat = G^dagger Z_S G`` (a single +-Pauli string), computed
    via ``G.to_tableau().inverse()(Z_S)``;
  * prepare the per-qubit ``+1`` eigenstate of ``S_hat`` (product of stabiliser states);
  * apply ``G`` with **server-side depolarising noise** on every gate;
  * measure all qubits; the round **fails** iff ``parity(outcomes over S) XOR sign(S_hat)``
    is 1 (noise flipped a useful outcome).

CSV columns ``p_depol,n,t,p_failed_round,p_false_reject`` mirror the MBQC benchmark's
``p_ent,width,depth,...`` (rename only). One CSV per ``(p_depol, shots)``; resume is per file.

Usage -- local (LocalCluster, all cores)
----------------------------------------
    python applications/benchmark-stim-msi/benchmark_msi_dask.py --ns 2,3,4 --ts 1,2,3

Usage -- SLURM cluster
----------------------
    python applications/benchmark-stim-msi/benchmark_msi_dask.py \\
        --ns 4,6,8 --ts 4,8,12 --depols 1e-3 \\
        --walltime 2 --memory 8 --cores 4 --port 8787 --scale 20
"""

from __future__ import annotations

import csv
import logging
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
from numpy.random import PCG64, Generator

app = typer.Typer(add_completion=False)

# Dask logs a benign CommClosedError from the heartbeat coroutine during teardown.
logging.getLogger("distributed.comm").setLevel(logging.CRITICAL)
logging.getLogger("distributed.client").setLevel(logging.CRITICAL)

# Frozen to mirror the MBQC benchmark (p_ent,width,depth -> p_depol,n,t). Shot count is
# encoded in the filename, not a column.
CSV_FIELDS = ["p_depol", "n", "t", "p_failed_round", "p_false_reject"]

# Single-qubit Clifford generators used to build random layers reproducibly from a numpy
# Generator (stim.Tableau.random is not seedable, which the Dask resume model requires).
_1Q = ("H", "S")
_2Q_GATES = frozenset({"CX", "CY", "CZ", "XCX", "XCZ", "YCX", "YCZ", "SWAP", "ISWAP"})

# Per-qubit prep gates (from |0>) for the +1 eigenstate of each Pauli letter, matching
# veriphix.verifying.generate_eigenstate (coin=0): I,Z -> |0>; X -> |+>; Y -> |+i>.
_PREP = {0: (), 1: ("H",), 2: ("H", "S"), 3: ()}


# ── circuit construction ─────────────────────────────────────────────────────────


def build_clifford_msi(n: int, t: int, clifford_depth: int, rng: Generator) -> tuple[stim.Circuit, int]:
    """Build the noiseless Clifford+MSI unitary ``G`` on ``N = n + t`` qubits.

    Data qubits are ``0..n-1``; the ``i``-th injection ancilla is qubit ``n+i``. Each
    Clifford layer ``C_i`` is a depth-``clifford_depth`` brickwork of random single-qubit
    Cliffords (words over ``{H,S}``) and ``CX`` gates on alternating neighbour pairs. Each
    gadget ``F_i = SWAP_{n+i,n-1} . CNOT_{n+i,n-1}`` follows its Clifford layer; the ancilla
    measurement is *deferred* (we measure everything at the end), so no ``M`` is emitted here.
    """
    circuit = stim.Circuit()
    data = list(range(n))

    def clifford_layer() -> None:
        for d in range(clifford_depth):
            for q in data:
                for _ in range(2):  # short random word -> well-mixed 1q Clifford
                    circuit.append(_1Q[int(rng.integers(2))], [q])
            for i in range(d % 2, n - 1, 2):  # brickwork CX entangling layer
                a, b = (data[i], data[i + 1]) if rng.random() < 0.5 else (data[i + 1], data[i])
                circuit.append("CX", [a, b])

    for i in range(t):
        clifford_layer()  # C_i
        anc = n + i
        circuit.append("CX", [anc, n - 1])  # F_i = SWAP . CNOT
        circuit.append("SWAP", [anc, n - 1])
    clifford_layer()  # C_{t+1}
    return circuit, n + t


def add_depolarising_noise(g_circuit: stim.Circuit, p_depol: float) -> stim.Circuit:
    """Return a copy of ``G`` with server-side depolarising noise after every gate.

    ``DEPOLARIZE1(p)`` after each single-qubit gate, ``DEPOLARIZE2(p)`` after each
    two-qubit gate. The Client's state prep and the final measurement stay noiseless
    (the Client is assumed perfect; noise models the Server applying ``G``).
    """
    out = stim.Circuit()
    for inst in g_circuit:
        out.append(inst)
        targets = [tg.value for tg in inst.targets_copy()]
        if inst.name in _2Q_GATES:
            for i in range(0, len(targets), 2):
                out.append("DEPOLARIZE2", [targets[i], targets[i + 1]], p_depol)
        else:
            for q in targets:
                out.append("DEPOLARIZE1", [q], p_depol)
    return out


def _sample_subsets(n_total: int, n_qubits: int, rng: Generator) -> np.ndarray:
    """``n_total`` independent **uniform** non-empty subsets of ``[n_qubits]``.

    Returns a boolean ``(n_total, n_qubits)`` membership matrix. Each row is drawn by
    flipping ``n_qubits`` fair coins; the all-zeros (empty) outcome is *resampled* (true
    rejection), so the distribution is exactly uniform over the ``2^N - 1`` non-empty
    subsets -- not biased toward singletons (which forcing a single bit would do).
    """
    subsets = rng.integers(0, 2, size=(n_total, n_qubits), dtype=bool)
    empty = np.flatnonzero(~subsets.any(axis=1))
    while empty.size:
        subsets[empty] = rng.integers(0, 2, size=(empty.size, n_qubits), dtype=bool)
        empty = empty[~subsets[empty].any(axis=1)]
    return subsets


def _round_fails(
    inv_tableau: stim.Tableau, noisy_g: stim.Circuit, n_qubits: int, n_total: int, rng: Generator
) -> tuple[np.ndarray, int]:
    """Faithful RandomTraps: a fresh uniform trap per round, exact (no reused pool).

    Draws ``n_total`` independent uniform subsets (one per test round), then **groups
    identical subsets** so each distinct trap circuit is compiled and sampled only once --
    the result is exactly fresh-per-round uniform sampling, with compilation cost equal to
    the number of *distinct* subsets drawn (~``n_total`` for large ``n+t``, fewer when
    collisions are likely). Returns the per-round failure flags and the distinct-trap count.
    """
    subsets = _sample_subsets(n_total, n_qubits, rng)
    keys = np.packbits(subsets, axis=1)  # one byte-key per row for grouping
    groups: dict[bytes, list[int]] = {}
    for i in range(n_total):
        groups.setdefault(keys[i].tobytes(), []).append(i)

    fails = np.empty(n_total, dtype=bool)
    for idxs in groups.values():
        trap = np.flatnonzero(subsets[idxs[0]]).tolist()
        fails[idxs] = _trap_fail_pool(inv_tableau, noisy_g, n_qubits, trap, len(idxs))
    return fails, len(groups)


def _trap_fail_pool(
    inv_tableau: stim.Tableau,
    noisy_g: stim.Circuit,
    n_qubits: int,
    trap: list[int],
    n_shots: int,
) -> np.ndarray:
    """Batch-sample ``n_shots`` honest-but-noisy rounds of one trap ``S``.

    Returns a boolean array ``(n_shots,)``: ``True`` where the round fails, i.e. where the
    measured parity over ``S`` (XOR the trap-stabiliser sign) is 1.
    """
    z_s = stim.PauliString(n_qubits)
    for q in trap:
        z_s[q] = 3  # Z
    stab = inv_tableau(z_s)  # S_hat = G^dagger Z_S G  (single +-Pauli string)
    sign_flip = int(stab.sign.real < 0)

    circuit = stim.Circuit()
    for q in range(n_qubits):  # prepare +1 eigenstate of S_hat (product state)
        for gate in _PREP[stab[q]]:
            circuit.append(gate, [q])
    circuit += noisy_g
    circuit.append("M", list(range(n_qubits)))  # all qubits, record index == qubit index

    samples = np.asarray(circuit.compile_sampler().sample(shots=n_shots))
    parity = (samples[:, trap].sum(axis=1) & 1) ^ sign_flip
    return parity.astype(bool)


# ── result types ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellResult:
    p_depol: float
    n: int
    t: int
    p_failed_round: float
    p_false_reject: float
    qubits: int = 0
    gates: int = 0
    traps_compiled: int = 0
    build_s: float = 0.0
    sample_s: float = 0.0
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class CellFailure:
    p_depol: float
    n: int
    t: int
    error: str
    elapsed_s: float = 0.0


# ── work unit ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """One ``(n, t, p_depol)`` tile -- the unit of Dask parallelism.

    Carries only serialisable scalars; the worker regenerates the fixed Clifford+MSI
    circuit and the traps deterministically from ``base_seed`` via
    ``PCG64(base_seed).jumped(n*1009 + t)`` -- so the same circuit/traps are reused across
    noise levels, and only ``p_depol`` changes between cells of the same ``(n, t)``.
    """

    n: int
    t: int
    p_depol: float
    n_shots: int
    test_rounds: int
    threshold: int
    clifford_depth: int
    n_traps: int
    base_seed: int

    def execute(self) -> CellResult | CellFailure:
        t0 = time.monotonic()
        try:
            t_b0 = time.monotonic()
            rng = Generator(PCG64(self.base_seed).jumped(self.n * 1009 + self.t))
            g_circuit, n_qubits = build_clifford_msi(self.n, self.t, self.clifford_depth, rng)
            inv_tableau = g_circuit.to_tableau().inverse()
            noisy_g = add_depolarising_noise(g_circuit, self.p_depol)
            build_s = time.monotonic() - t_b0

            t_s0 = time.monotonic()
            n_total = self.n_shots * self.test_rounds
            if self.n_traps <= 0:
                # Default: exact fresh-per-round uniform RandomTraps.
                flat, traps_compiled = _round_fails(inv_tableau, noisy_g, n_qubits, n_total, rng)
            else:
                # Opt-in fast approximation: draw a fixed pool of n_traps uniform subsets and
                # assign each round one uniformly. Unbiased for p_failed_round; only an
                # approximation for p_false_reject (finite pool correlates rounds within a shot).
                pool = _sample_subsets(self.n_traps, n_qubits, rng)
                choice = rng.integers(0, self.n_traps, size=n_total)
                flat = np.empty(n_total, dtype=bool)
                for k in range(self.n_traps):
                    idxs = np.flatnonzero(choice == k)
                    if idxs.size:
                        trap = np.flatnonzero(pool[k]).tolist()
                        flat[idxs] = _trap_fail_pool(inv_tableau, noisy_g, n_qubits, trap, idxs.size)
                traps_compiled = self.n_traps
            fails = flat.reshape(self.n_shots, self.test_rounds)
            sample_s = time.monotonic() - t_s0

            nr_failed = fails.sum(axis=1)
            return CellResult(
                p_depol=self.p_depol,
                n=self.n,
                t=self.t,
                p_failed_round=float(fails.mean()),
                p_false_reject=float((nr_failed > self.threshold).mean()),
                qubits=n_qubits,
                gates=len(g_circuit),
                traps_compiled=traps_compiled,
                build_s=build_s,
                sample_s=sample_s,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
            return CellFailure(
                p_depol=self.p_depol, n=self.n, t=self.t, error=str(exc), elapsed_s=time.monotonic() - t0
            )


# ── cluster helpers (mirror benchmark_stim_dask.py) ─────────────────────────────────


def _get_cluster(
    walltime: int | None, memory: int | None, cores: int | None, port: int | None, scale: int | None
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
        )
    if scale is not None:
        cluster.scale(scale)
    return cluster


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    """The set of (p_depol, n, t) already present in the CSV (for resume)."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        return {(row["p_depol"], row["n"], row["t"]) for row in csv.DictReader(f)}


def _csv_path(out_dir: Path, p_depol: float, shots: int) -> Path:
    return out_dir / f"benchmark_msi_results_p{p_depol:.1e}_s{shots}.csv"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.2f}s"


# Per-tile cost grows roughly with the qubit count times the trap-pool work.
def _cost(c: Cell) -> int:
    return (c.n + c.t) * c.n_traps


# ── main ────────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    ns:            Annotated[str, typer.Option(help="Comma-separated data-qubit counts n")] = "4,6,8",
    ts:            Annotated[str, typer.Option(help="Comma-separated injection-layer counts t")] = "4,8,12",
    depols:        Annotated[str, typer.Option(help="Comma-separated depolarising probs")] = "1e-3",
    shots:         Annotated[int, typer.Option(help="Verification instances per cell")] = 100,
    test_rounds:   Annotated[int, typer.Option(help="Test rounds per instance")] = 100,
    threshold:     Annotated[int, typer.Option(help="Tolerated failed test rounds (w)")] = 0,
    clifford_depth: Annotated[int, typer.Option(help="Brickwork depth of each Clifford layer C_i")] = 2,
    n_traps:       Annotated[int, typer.Option(help="0 = exact fresh-per-round uniform traps (default); "
                                                    ">0 = fast pooled approximation with that many traps")] = 0,
    out_dir:       Annotated[Path, typer.Option(help="Directory for per-(p_depol,shots) CSVs")] = Path("applications/benchmark-stim-msi"),
    seed:          Annotated[int, typer.Option()] = 42,
    walltime:      Annotated[int | None, typer.Option(help="SLURM: walltime in hours")] = None,
    memory:        Annotated[int | None, typer.Option(help="SLURM: memory in GB")] = None,
    cores:         Annotated[int | None, typer.Option(help="SLURM: cores per job")] = None,
    port:          Annotated[int | None, typer.Option(help="SLURM: dashboard port")] = None,
    scale:         Annotated[int | None, typer.Option(help="Number of workers")] = None,
    smoke:         Annotated[bool, typer.Option()] = False,
) -> None:
    """Sweep (n, t) x p_depol across a Dask cluster; write the honest-failure CSV(s)."""
    if smoke:
        ns, ts, depols, shots, test_rounds = "2,3", "1,2", "1e-2", 20, 20

    n_list = _parse_ints(ns)
    t_list = _parse_ints(ts)
    depol_list = _parse_floats(depols)

    # One Cell per (n, t, p_depol); n fastest within a t (mirrors the MBQC driver order).
    dims = [(n, t) for t in t_list for n in n_list]
    cells = [
        Cell(
            n=n, t=t, p_depol=p, n_shots=shots, test_rounds=test_rounds,
            threshold=threshold, clifford_depth=clifford_depth, n_traps=n_traps, base_seed=seed,
        )
        for (n, t) in dims
        for p in depol_list
    ]
    n_cells_total = len(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {p: _csv_path(out_dir, p, shots) for p in depol_list}
    done = {p: _load_done(path) for p, path in paths.items()}
    n_existing = sum(len(d) for d in done.values())
    if n_existing:
        typer.echo(f"Resuming: {n_existing} cells already on disk across {len(paths)} file(s)")
    cells = [c for c in cells if (str(c.p_depol), str(c.n), str(c.t)) not in done[c.p_depol]]

    # Largest tiles first (LPT) so the cheap ones backfill idle workers at the tail.
    cells.sort(key=_cost, reverse=True)

    typer.echo(
        f"grid: {len(n_list)} n x {len(t_list)} t x {len(depol_list)} noise levels "
        f"= {n_cells_total} cells ({len(cells)} to run); "
        f"shots={shots}, test_rounds={test_rounds}, n_traps={n_traps}, clifford_depth={clifford_depth}"
    )
    if not cells:
        typer.echo("Nothing to do -- all cells already present.")
        return

    cluster = _get_cluster(walltime, memory, cores, port, scale)
    dask_client = dask.distributed.Client(cluster)
    typer.echo(f"Dask dashboard: {dask_client.dashboard_link}")

    n_ok = n_fail = 0
    loop_start = time.monotonic()
    try:
        futures = [dask_client.submit(Cell.execute, c, pure=False) for c in cells]

        writers: dict[float, tuple[object, csv.DictWriter]] = {}
        try:
            for p, path in paths.items():
                is_new = not path.exists() or path.stat().st_size == 0
                fh = path.open("a", newline="")
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                if is_new:
                    writer.writeheader()
                writers[p] = (fh, writer)

            for fut in dask.distributed.as_completed(futures):
                try:
                    report = fut.result()
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"Future error: {exc}")
                    n_fail += 1
                    continue
                finally:
                    fut.release()

                done_count = n_ok + n_fail + 1
                elapsed = time.monotonic() - loop_start
                eta = elapsed / done_count * (len(cells) - done_count)

                if isinstance(report, CellResult):
                    fh, writer = writers[report.p_depol]
                    writer.writerow(
                        {
                            "p_depol": report.p_depol,
                            "n": report.n,
                            "t": report.t,
                            "p_failed_round": report.p_failed_round,
                            "p_false_reject": report.p_false_reject,
                        }
                    )
                    fh.flush()
                    n_ok += 1
                    typer.echo(
                        f"  [{n_ok + n_fail}/{len(cells)}] n={report.n:>2} t={report.t:>2} "
                        f"p={report.p_depol:.1e} qubits={report.qubits:>3} gates={report.gates:>5} "
                        f"traps={report.traps_compiled:>5}  "
                        f"build={report.build_s:.2f}s sample={report.sample_s:.2f}s "
                        f"cell={_fmt(report.elapsed_s)}  "
                        f"p_fail_round={report.p_failed_round:.4f} "
                        f"p_false_reject={report.p_false_reject:.3f}  ETA {_fmt(eta)}"
                    )
                else:
                    n_fail += 1
                    typer.echo(
                        f"  x [{n_ok + n_fail}/{len(cells)}] n={report.n:>2} t={report.t:>2} "
                        f"p={report.p_depol:.1e}  t={report.elapsed_s:.1f}s  ETA {_fmt(eta)}: {report.error}"
                    )
        finally:
            for fh, _ in writers.values():
                fh.close()
    finally:
        dask_client.close()
        cluster.close()

    typer.echo(
        f"\nDone. {n_ok} results, {n_fail} failures  "
        f"(wall {_fmt(time.monotonic() - loop_start)})  ->  {len(paths)} file(s) in {out_dir}"
    )
    for p in sorted(paths):
        typer.echo(f"    p_depol={p:.1e} -> {paths[p].name}")


if __name__ == "__main__":
    freeze_support()
    app()
