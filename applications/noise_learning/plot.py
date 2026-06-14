"""plot.py — three-panel comparison of observed vs. true noise.

Panel 1 (observed)      : FK12 trap-failure rate per node, averaged over all
                          circuit CSVs — the *client-side* view of where things
                          failed (same style as hotgate).
Panel 2 (ground truth)  : the exact per-node flip probability of the noise model
                          (from ``ground_truth.csv``).
Panel 3 (difference)    : observed − ground truth, diverging colormap centred
                          at 0 — how far the client's reconstruction is.

Note: panel 1 is a *trap-failure rate* and panel 2 a *flip probability*; a
triggered dephasing flip does not deterministically flip a trap outcome, so
panel 3 is a qualitative comparison of where the noise shows up, not a strict
calibration error.

Usage
-----
    python applications/noise_learning/plot.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import Normalize, TwoSlopeNorm
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12, get_bipartite_coloring, get_node_positions
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.verifying import TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)

SAMPLED_BASE = Path("applications/gospel/sampled_circuits")


def _load_graph_and_positions(circuits_dir: Path):
    """Load the brickwork graph and node grid positions from the first circuit."""
    qasm_files = sorted(circuits_dir.glob("*.qasm"))
    if not qasm_files:
        raise RuntimeError(f"No .qasm files found in {circuits_dir}")
    with qasm_files[0].open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()

    node_positions = {
        node: (int(pos[0]), int(pos[1]))
        for node, pos in get_node_positions(pattern).items()
    }
    red, blue = get_bipartite_coloring(pattern)
    client = Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=FK12(manual_colouring=(red, blue)),
        parameters=TrappifiedSchemeParameters(comp_rounds=0, test_rounds=1, threshold=0),
    )
    return client.graph, node_positions


def _draw_panel(ax, graph, node_positions, values, *, cmap, norm, title, cbar_label):
    """Draw the brickwork graph with nodes coloured by ``values[(col,row)]``."""
    radius = 0.38
    for u, v in graph.edges():
        cu, ru = node_positions[u]
        cv, rv = node_positions[v]
        ax.plot([cu, cv], [ru, rv], "k-", linewidth=0.6, zorder=1)
    for node, (col, row) in node_positions.items():
        color = cmap(norm(values.get((col, row), 0.0)))
        ax.add_patch(mpatches.Circle((col, row), radius, color=color, zorder=2))

    n_cols = max(c for c, _ in node_positions.values()) + 1
    n_rows = max(r for _, r in node_positions.values()) + 1
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, label=cbar_label)


@app.command()
def main(
    results_dir:   Annotated[Path, typer.Option(help="Directory with per-circuit CSVs + ground_truth.csv")] = Path("applications/noise_learning/results"),
    out:           Annotated[Path, typer.Option(help="Output PDF path")]                                     = Path("applications/noise_learning/plots/comparison.pdf"),
    n_qubits:      Annotated[int,  typer.Option(help="Number of qubits")]                                     = 3,
    depth:         Annotated[int,  typer.Option(help="Circuit depth")]                                        = 5,
    bqp_error:     Annotated[str,  typer.Option(help="BQP error tag")]                                        = "1e-1",
    n_test_rounds: Annotated[int,  typer.Option(help="Test rounds label for the title")]                     = 100,
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"

    # ── observed failure rate, aggregated over all circuit CSVs ────────────────
    csv_files = sorted(results_dir.glob("circuit_*.csv"))
    if not csv_files:
        typer.echo(f"[ERROR] No circuit_*.csv files found in {results_dir}")
        raise typer.Exit(1)

    failure_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    total_tests:    defaultdict[tuple[int, int], int] = defaultdict(int)
    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                pos = (int(row["col"]), int(row["row"]))
                failure_counts[pos] += int(row["failure_count"])
                total_tests[pos]    += int(row["total_tests"])
    observed = {
        pos: (failure_counts[pos] / total_tests[pos] if total_tests[pos] else 0.0)
        for pos in total_tests
    }

    # ── ground-truth flip probabilities ───────────────────────────────────────
    gt_csv = results_dir / "ground_truth.csv"
    if not gt_csv.exists():
        typer.echo(f"[ERROR] {gt_csv} not found — run simulate.py first.")
        raise typer.Exit(1)
    ground_truth: dict[tuple[int, int], float] = {}
    with gt_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            ground_truth[(int(row["col"]), int(row["row"]))] = float(row["prob"])

    difference = {pos: observed.get(pos, 0.0) - ground_truth.get(pos, 0.0) for pos in ground_truth}

    graph, node_positions = _load_graph_and_positions(circuits_dir)
    n_circuits = len(csv_files)

    # ── three side-by-side panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    rate_cmap = plt.colormaps["jet"]
    rate_norm = Normalize(vmin=0.0, vmax=1.0)
    _draw_panel(
        axes[0], graph, node_positions, observed,
        cmap=rate_cmap, norm=rate_norm,
        title=f"Observed (FK12 trap failure rate)\n{n_circuits} circuits × {n_test_rounds} rounds",
        cbar_label="failure rate",
    )
    _draw_panel(
        axes[1], graph, node_positions, ground_truth,
        cmap=rate_cmap, norm=rate_norm,
        title="Ground truth (flip probability)",
        cbar_label="flip prob",
    )

    max_abs = max((abs(d) for d in difference.values()), default=1.0) or 1.0
    diff_cmap = plt.colormaps["RdBu_r"]
    diff_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    _draw_panel(
        axes[2], graph, node_positions, difference,
        cmap=diff_cmap, norm=diff_norm,
        title="Difference (observed − ground truth)",
        cbar_label="observed − truth",
    )

    fig.suptitle(
        f"Noise-learning: client reconstruction vs. true noise  (n={n_qubits}, d={depth})",
        fontsize=13,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    typer.echo(f"Saved → {out}")


if __name__ == "__main__":
    app()
