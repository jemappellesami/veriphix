"""Stim/Clifford clone of the verification benchmark.

Same honest-failure / feasibility-region pipeline as ``applications/benchmarking.py``
and the gospel benchmark, but everything is turned Clifford so it runs in **Stim** in
polynomial time and reaches sizes the density-matrix backends cannot.

Deltas vs the original benchmark
--------------------------------
* computation : random-Pauli **brickwork** (Clifford by construction), (width, depth) = (nqubits, nlayers)
* noise       : ACES :class:`UncorrelatedDepolarisingNoiseModel` (1-qubit depol per CZ endpoint)
* backend     : **Stim** via ``pattern_to_stim_circuit`` (batched sampling)
* blinding    : off (theta blinding would make measurements non-Clifford; the *honest*
                failure probability is blinding-invariant anyway)
* Monte Carlo : ONE fixed circuit per cell, sampled ``shots * test_rounds`` times,
                folded into ``shots`` verification instances of ``test_rounds`` rounds each
* traps       : FK12 bipartite (2 fixed test runs -> the whole round table is two batched
                ``sample()`` calls)

Output CSV columns ``p_ent,width,depth,p_failed_round,p_false_reject`` are identical to
``benchmarking.py``, so ``applications/plot_veriphix_heatmaps.py`` plots it unchanged.

Usage
-----
    python applications/benchmark-stim/benchmark_stim.py --widths 2,4,6 --depths 2,4,8
    python applications/benchmark-stim/benchmark_stim.py --smoke

The run prints a per-cell timing line and, at the end, a **per-dimension timing table**
so you can see what is feasible on a single machine without a cluster.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import typer
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

# The Stim transpiler is vendored in the ACES experiment folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "aces"))
from stim_pauli_preprocessing import BasicState, pattern_to_stim_circuit

app = typer.Typer(add_completion=False)


# ── helpers (mirror the ACES port) ──────────────────────────────────────────────


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


def simulate_cell(
    width: int,
    depth: int,
    p_ent: float,
    n_shots: int,
    test_rounds: int,
    threshold: int,
    base_seed: int,
) -> tuple[float, float, dict[str, float]]:
    """Simulate one (width, depth, p_ent) tile. Returns (p_failed_round, p_false_reject, timings)."""
    t0 = time.monotonic()
    # Deterministic per (width, depth): the same fixed circuit is reused across noise levels.
    rng = Generator(PCG64(base_seed).jumped(width * 1009 + depth))
    pattern = generate_random_pauli_pattern(
        nqubits=width, nlayers=depth, order=ConstructionOrder.Canonical, rng=rng
    )
    colours = get_bipartite_coloring(pattern)
    client = Client(
        pattern=pattern,
        secrets=Secrets(r=False, a=False, theta=False),
        protocol=FK12(manual_colouring=list(colours)),
        rng=rng,
    )
    stim_pattern = x_basis_measurement_pattern(client.clean_pattern)
    test_runs = client.test_runs
    t_build = time.monotonic() - t0

    t0 = time.monotonic()
    n_total = n_shots * test_rounds
    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=p_ent)
    pools = [_round_fail_pool(run, stim_pattern, noise_model, n_total) for run in test_runs]

    # Each round independently picks a test run (FK12.sample_test_run is uniform).
    choice = rng.integers(0, len(pools), size=n_total)
    fails = np.empty(n_total, dtype=bool)
    for k, pool in enumerate(pools):
        mask = choice == k
        fails[mask] = pool[: int(mask.sum())]
    fails = fails.reshape(n_shots, test_rounds)
    t_sample = time.monotonic() - t0

    nr_failed = fails.sum(axis=1)
    p_failed_round = float(fails.mean())
    p_false_reject = float((nr_failed > threshold).mean())
    timings = {
        "build": t_build,
        "sample": t_sample,
        "nodes": float(pattern.n_node),
        "edges": float(sum(1 for c in pattern if c.kind == CommandKind.E)),
    }
    return p_failed_round, p_false_reject, timings


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s" if m else f"{s:.2f}s"


@app.command()
def main(
    widths: str = "8, 9, 10",
    depths: str = "16",
    ent_errors: str = "1e-3",
    shots: int = 100,
    test_rounds: int = 100,
    threshold: int = 0,
    out_csv: Path = Path("applications/benchmark-stim/benchmark_stim_results.csv"),
    seed: int = 42,
    max_cell_seconds: float = 0.0,
    smoke: bool = False,
) -> None:
    """Sweep (width, depth) x p_ent; write the honest-failure landscape CSV with timing."""
    if smoke:
        widths, depths, ent_errors, shots, test_rounds = "2,3", "2,3", "1e-2", 20, 20

    width_list = _parse_ints(widths)
    depth_list = _parse_ints(depths)
    ent_list = _parse_floats(ent_errors)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_csv.exists() or out_csv.stat().st_size == 0
    fh = out_csv.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=["p_ent", "width", "depth", "p_failed_round", "p_false_reject"])
    if is_new:
        writer.writeheader()

    dims = [(w, d) for d in depth_list for w in width_list]
    n_cells = len(dims) * len(ent_list)
    typer.echo(
        f"grid: {len(width_list)} widths x {len(depth_list)} depths x {len(ent_list)} noise levels "
        f"= {n_cells} cells; shots={shots}, test_rounds={test_rounds} ({shots * test_rounds} samples/colour)"
    )

    dim_times: dict[tuple[int, int], dict[str, float]] = {}
    cell_idx = 0
    t_run = time.monotonic()
    try:
        for width, depth in dims:
            t_dim = time.monotonic()
            build_t = sample_t = 0.0
            nodes = edges = 0.0
            for p_ent in ent_list:
                cell_idx += 1
                p_failed_round, p_false_reject, tm = simulate_cell(
                    width, depth, p_ent, shots, test_rounds, threshold, seed
                )
                build_t += tm["build"]
                sample_t += tm["sample"]
                nodes, edges = tm["nodes"], tm["edges"]
                writer.writerow(
                    {
                        "p_ent": p_ent,
                        "width": width,
                        "depth": depth,
                        "p_failed_round": p_failed_round,
                        "p_false_reject": p_false_reject,
                    }
                )
                fh.flush()
                cell_t = tm["build"] + tm["sample"]
                eta = (time.monotonic() - t_run) / cell_idx * (n_cells - cell_idx)
                typer.echo(
                    f"  [{cell_idx}/{n_cells}] w={width:>2} d={depth:>2} p={p_ent:.1e} "
                    f"|V|={int(nodes):>4} |E|={int(edges):>4}  "
                    f"build={tm['build']:.2f}s sample={tm['sample']:.2f}s cell={_fmt(cell_t)}  "
                    f"p_fail_round={p_failed_round:.4f} p_false_reject={p_false_reject:.3f}  ETA {_fmt(eta)}"
                )
            dim_t = time.monotonic() - t_dim
            dim_times[(width, depth)] = {
                "nodes": nodes,
                "edges": edges,
                "build": build_t,
                "sample": sample_t,
                "total": dim_t,
                "per_cell": dim_t / len(ent_list),
            }
    finally:
        fh.close()

    # ── per-dimension timing summary ───────────────────────────────────────────
    typer.echo("\n" + "=" * 96)
    typer.echo("PER-DIMENSION TIMING  (one cell = one noise level; a full noise sweep = per_cell x #levels)")
    typer.echo("=" * 96)
    typer.echo(f"{'width':>5} {'depth':>5} {'|V|':>6} {'|E|':>6} {'build/cell':>11} {'sample/cell':>12} {'per_cell':>10}")
    for (width, depth), t in sorted(dim_times.items(), key=lambda kv: kv[1]["nodes"]):
        n_lvl = len(ent_list)
        typer.echo(
            f"{width:>5} {depth:>5} {int(t['nodes']):>6} {int(t['edges']):>6} "
            f"{t['build'] / n_lvl:>10.2f}s {t['sample'] / n_lvl:>11.2f}s {t['per_cell']:>9.2f}s"
        )
    typer.echo(f"\ntotal wall time: {_fmt(time.monotonic() - t_run)}  ->  {out_csv}")


if __name__ == "__main__":
    app()
