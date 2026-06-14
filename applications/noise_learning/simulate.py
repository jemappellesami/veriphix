"""simulate.py — FK12 trap-failure under stochastic Gaussian-region noise.

Runs FK12 verification on the n=3, d=5 sampled circuits using a single, shared
:class:`GaussianRegionNoiseModel`.  The noisy region(s) are defined *once*
(graph BFS + Gaussian, see ``regions.py``) against the fixed brickwork scaffold
and reused for every circuit — only the per-circuit node-id remapping differs.

For each circuit it records, per trap node, how many test rounds failed, in the
same CSV format as ``applications/gospel/hotgate``.  The ground-truth flip
probabilities are written once to ``ground_truth.csv`` for the plotting step.

Usage
-----
    python applications/noise_learning/simulate.py
    python applications/noise_learning/simulate.py --n-circuits 10 --n-test-rounds 100
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import typer
from graphix.sim.density_matrix import DensityMatrixBackend
from tqdm import tqdm
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.gaussian_region_noise_model import GaussianRegionNoiseModel
from veriphix.protocols import FK12, get_bipartite_coloring, get_node_positions
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TestResult, TrappifiedSchemeParameters

from regions import RegionSpec, build_node_probs

app = typer.Typer(add_completion=False)

CSV_HEADER = ["node", "col", "row", "failure_count", "total_tests"]
GROUND_TRUTH_HEADER = ["node", "col", "row", "prob"]

SAMPLED_BASE = Path("applications/gospel/sampled_circuits")

# ── region specification ──────────────────────────────────────────────────────
# Multi-region noise: several Gaussian hot-spots spread along the n=3, d=5
# brickwork strip (63 nodes, cols 0..20, rows 0..2).  Centres sit on different
# rows and the two central regions (nodes 24 & 31) overlap, so their summed
# probability clips toward 1.0 — a genuinely complex, uneven noise landscape.
# Expressed in terms of the *canonical* node ids of the first circuit's graph.
REGION_SPECS: list[RegionSpec] = [
    RegionSpec(center=7,  depth=2, sigma=1.2, peak=0.50),  # left blob (col 2, row 1)
    RegionSpec(center=24, depth=2, sigma=1.3, peak=0.45),  # mid-left (col 8, row 0)
    RegionSpec(center=31, depth=2, sigma=1.5, peak=0.70),  # central, overlaps 24 (col 10, row 1)
    RegionSpec(center=47, depth=1, sigma=0.8, peak=0.55),  # sharp spot (col 15, row 2)
    RegionSpec(center=55, depth=2, sigma=1.0, peak=0.40),  # right, weaker (col 18, row 1)
]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _load_pattern(path: Path):
    with path.open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    return pattern


def _make_client(pattern, rng, n_test_rounds: int) -> Client:
    red, blue = get_bipartite_coloring(pattern)
    return Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=FK12(manual_colouring=(red, blue)),
        parameters=TrappifiedSchemeParameters(
            comp_rounds=0, test_rounds=n_test_rounds, threshold=0
        ),
        rng=rng,
    )


def _delegate_with_progress(client, canvas, noise_model, rng, desc):
    """Mirror of ``Client.delegate_canvas`` with a per-round tqdm bar.

    Replicated here (rather than calling ``delegate_canvas`` directly) so each
    test round updates the progress bar live — the library call runs all rounds
    in one opaque pass.  Kept deliberately minimal; the only API touched is the
    stable ``Run.accept`` contract.
    """
    outcomes = {}
    for r in tqdm(canvas, desc=desc, unit="round", leave=False):
        backend = DensityMatrixBackend()
        outcomes[r] = canvas[r].accept(client, backend, noise_model, rng)
    return outcomes


@app.command()
def main(
    n_qubits:      Annotated[int,   typer.Option(help="Number of qubits")]                          = 3,
    depth:         Annotated[int,   typer.Option(help="Circuit depth")]                             = 5,
    bqp_error:     Annotated[str,   typer.Option(help="BQP error tag (folder suffix)")]             = "1e-1",
    n_circuits:    Annotated[int,   typer.Option(help="Number of circuits to average over")]        = 10,
    n_test_rounds: Annotated[int,   typer.Option(help="Test rounds per circuit")]                   = 100,
    base_seed:     Annotated[int,   typer.Option(help="Base RNG seed (seed = base_seed + idx)")]    = 42,
    out_dir:       Annotated[Path,  typer.Option(help="Directory for per-circuit CSVs")]            = Path("applications/noise_learning/results"),
    force:         Annotated[bool,  typer.Option(help="Re-run circuits even if CSV exists")]        = False,
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"
    if not circuits_dir.exists():
        typer.echo(f"[ERROR] circuits directory not found: {circuits_dir}")
        raise typer.Exit(1)

    circuit_files = sorted(circuits_dir.glob("*.qasm"))[:n_circuits]
    if len(circuit_files) < n_circuits:
        typer.echo(f"[WARN] only {len(circuit_files)} circuits available (requested {n_circuits})")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── build the canonical noise map ONCE from the first circuit ──────────────
    ref_pattern = _load_pattern(circuit_files[0])
    ref_client = _make_client(ref_pattern, np.random.default_rng(base_seed), n_test_rounds)
    ref_positions = {
        node: (int(pos[0]), int(pos[1]))
        for node, pos in get_node_positions(ref_pattern).items()
    }

    canonical_probs = build_node_probs(ref_client.graph, REGION_SPECS)
    # Key the ground truth by (col, row) so it can be remapped onto each circuit.
    probs_by_pos: dict[tuple[int, int], float] = {
        ref_positions[node]: prob for node, prob in canonical_probs.items()
    }

    typer.echo(
        f"Noise regions {REGION_SPECS} → {len(canonical_probs)} noisy nodes "
        f"(max prob {max(canonical_probs.values()):.3f})"
    )

    # Persist ground truth once.
    gt_csv = out_dir / "ground_truth.csv"
    with gt_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GROUND_TRUTH_HEADER)
        writer.writeheader()
        for node, (col, row) in ref_positions.items():
            writer.writerow({
                "node": node, "col": col, "row": row,
                "prob": probs_by_pos.get((col, row), 0.0),
            })
    typer.echo(f"Ground truth → {gt_csv}")

    # ── per-circuit FK12 runs ──────────────────────────────────────────────────
    n_total = len(circuit_files)
    t_run_start = time.monotonic()
    n_done = 0  # circuits actually simulated this invocation (for ETA)

    for idx, circuit_path in enumerate(circuit_files):
        out_csv = out_dir / f"circuit_{idx:03d}.csv"
        if out_csv.exists() and not force:
            typer.echo(f"  [{idx+1}/{n_total}] skip (already done)")
            continue

        rng = np.random.default_rng(base_seed + idx)
        pattern = _load_pattern(circuit_path)
        node_positions = {
            node: (int(pos[0]), int(pos[1]))
            for node, pos in get_node_positions(pattern).items()
        }

        # Remap the shared noise map onto this circuit's node ids via (col, row).
        node_probs = {
            node: probs_by_pos[pos]
            for node, pos in node_positions.items()
            if pos in probs_by_pos
        }
        noise_model = GaussianRegionNoiseModel(node_probs, rng=rng)

        client = _make_client(pattern, rng, n_test_rounds)
        canvas = client.sample_canvas(rng=rng)

        t_circuit = time.monotonic()
        outcomes = _delegate_with_progress(
            client, canvas, noise_model, rng,
            desc=f"[{idx+1}/{n_total}] {circuit_path.name}",
        )

        failure_counts: dict[int, int] = {n: 0 for n in node_positions}
        total_tests:    dict[int, int] = {n: 0 for n in node_positions}
        for run_result in outcomes.values():
            if not isinstance(run_result, TestResult):
                continue
            for trap, outcome in run_result.trap_outcomes.items():
                (node,) = trap
                total_tests[node]    += 1
                failure_counts[node] += outcome

        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
            writer.writeheader()
            for node, (col, row) in node_positions.items():
                writer.writerow({
                    "node": node, "col": col, "row": row,
                    "failure_count": failure_counts[node],
                    "total_tests":   total_tests[node],
                })

        n_done += 1
        max_rate = max(
            (failure_counts[n] / total_tests[n] for n in node_positions if total_tests[n]),
            default=0.0,
        )
        circuit_time = time.monotonic() - t_circuit
        elapsed = time.monotonic() - t_run_start
        eta = (elapsed / n_done) * (n_total - (idx + 1))
        typer.echo(
            f"  [{idx+1}/{n_total}] {circuit_path.name} → {out_csv.name}  "
            f"max_rate={max_rate:.3f}  "
            f"circuit {_fmt_time(circuit_time)}  total {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}"
        )

    typer.echo(f"Done in {_fmt_time(time.monotonic() - t_run_start)}.")


if __name__ == "__main__":
    app()
