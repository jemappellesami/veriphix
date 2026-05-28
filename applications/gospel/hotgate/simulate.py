"""simulate.py — FK12 trap-failure worker (one circuit per SLURM task).

Each invocation processes a single circuit and writes per-node failure counts
to a CSV file.  Run with a SLURM array job (see submit.sh) or locally in a
simple loop.

Usage
-----
    python applications/gospel/hotgate/simulate.py --circuit-idx 0
    python applications/gospel/hotgate/simulate.py --circuit-idx $SLURM_ARRAY_TASK_ID \\
        --circuits-dir applications/gospel/circuits/circuits-3-6 \\
        --n-test-rounds 100 --p-ent 2e-3 --out-dir applications/gospel/hotgate/results
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import typer
from graphix.noise_models import DepolarisingNoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12, get_bipartite_coloring, get_node_positions
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TestResult, TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)

CSV_HEADER = ["node", "col", "row", "failure_count", "total_tests"]


def _load_pattern(path: Path):
    with path.open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    return pattern


@app.command()
def main(
    circuit_idx: Annotated[int,  typer.Option(help="Index of the circuit to process (0-based)")] = 0,
    circuits_dir: Annotated[Path, typer.Option(help="Directory containing .qasm circuit files")] = Path("applications/gospel/circuits/circuits-6-6"),
    n_test_rounds: Annotated[int,  typer.Option(help="Number of test rounds per circuit")]          = 100,
    p_ent: Annotated[float, typer.Option(help="Depolarising entanglement error probability")]  = 2e-3,
    base_seed: Annotated[int,  typer.Option(help="Base RNG seed (actual seed = base_seed + circuit_idx)")] = 42,
    out_dir: Annotated[Path, typer.Option(help="Directory for per-circuit CSV output")]         = Path("applications/gospel/hotgate/results"),
) -> None:
    circuit_files = sorted(circuits_dir.glob("*.qasm"))
    if circuit_idx >= len(circuit_files):
        typer.echo(f"[ERROR] circuit_idx={circuit_idx} out of range (only {len(circuit_files)} circuits)")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"circuit_{circuit_idx:03d}.csv"

    if out_csv.exists():
        typer.echo(f"[SKIP] {out_csv} already exists — delete it to rerun.")
        raise typer.Exit(0)

    circuit_path = circuit_files[circuit_idx]
    # Each circuit gets a unique seed derived from the base so jobs are reproducible
    # but independent.
    rng = np.random.default_rng(base_seed + circuit_idx)

    noise_model = DepolarisingNoiseModel(
        entanglement_error_prob=p_ent,
        measure_error_prob=0.0,
        x_error_prob=0.0,
        z_error_prob=0.0,
        measure_channel_prob=0.0,
    )

    t0 = time.monotonic()
    typer.echo(f"[{circuit_idx}] {circuit_path.name}  p_ent={p_ent:.1e}  rounds={n_test_rounds}")

    pattern = _load_pattern(circuit_path)

    node_positions = {
        node: (int(pos[0]), int(pos[1]))
        for node, pos in get_node_positions(pattern).items()
    }

    red, blue = get_bipartite_coloring(pattern)
    protocol  = FK12(manual_colouring=(red, blue))

    client = Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=protocol,
        parameters=TrappifiedSchemeParameters(
            comp_rounds=0, test_rounds=n_test_rounds, threshold=0
        ),
        rng=rng,
    )

    canvas   = client.sample_canvas(rng=rng)
    outcomes = client.delegate_canvas(
        canvas=canvas,
        backend_cls=DensityMatrixBackend,
        noise_model=noise_model,
        rng=rng,
    )

    # Accumulate per-node failure counts.
    failure_counts: dict[int, int] = {n: 0 for n in node_positions}
    total_tests:    dict[int, int] = {n: 0 for n in node_positions}

    for run_result in outcomes.values():
        if not isinstance(run_result, TestResult):
            continue
        for trap, outcome in run_result.trap_outcomes.items():
            (node,) = trap
            total_tests[node]    += 1
            failure_counts[node] += outcome

    # Write CSV.
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        for node, (col, row) in node_positions.items():
            writer.writerow({
                "node":          node,
                "col":           col,
                "row":           row,
                "failure_count": failure_counts[node],
                "total_tests":   total_tests[node],
            })

    elapsed = time.monotonic() - t0
    m, s = divmod(int(elapsed), 60)
    typer.echo(f"[{circuit_idx}] done in {m}m{s:02d}s → {out_csv}")


if __name__ == "__main__":
    app()
