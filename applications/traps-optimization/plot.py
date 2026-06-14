"""plot.py — visualise the optimised trap distribution and the LP game.

Figure 1 ("the game"), three graph panels side by side:
  1. Learned noise        : observed trap-failure rate per node (the client's input).
  2. Optimal trap distrib. : per-node probability of being tested under the LP
                             optimum — where the client steers its test budget.
  3. Adversary (LP dual)   : the optimal attack's weight per node — which errors
                             are hardest for the chosen tests to catch.

Figure 2: detection-rate comparison bar chart (OptimizedTraps vs baselines).

Usage
-----
    python applications/traps-optimization/plot.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import typer
from matplotlib.colors import Normalize
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


def _draw_panel(ax, graph, node_positions, values, *, cmap, vmax, title, cbar_label):
    norm = Normalize(vmin=0.0, vmax=vmax)
    radius = 0.38
    for u, v in graph.edges():
        cu, ru = node_positions[u]
        cv, rv = node_positions[v]
        ax.plot([cu, cv], [ru, rv], "k-", linewidth=0.6, zorder=1)
    for node, (col, row) in node_positions.items():
        ax.add_patch(mpatches.Circle((col, row), radius, color=cmap(norm(values.get(node, 0.0))), zorder=2))
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
    results_dir: Annotated[Path, typer.Option(help="Directory with node_data.csv + detection_rates.json")] = Path("applications/traps-optimization/results"),
    out_game:    Annotated[Path, typer.Option(help="Output PDF for the 3-panel game figure")]              = Path("applications/traps-optimization/plots/game.pdf"),
    out_rates:   Annotated[Path, typer.Option(help="Output PDF for the detection-rate bar chart")]         = Path("applications/traps-optimization/plots/detection_rates.pdf"),
    n_qubits:    Annotated[int,  typer.Option(help="Number of qubits")]                                    = 3,
    depth:       Annotated[int,  typer.Option(help="Circuit depth")]                                       = 5,
    bqp_error:   Annotated[str,  typer.Option(help="BQP error tag")]                                       = "1e-1",
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"
    graph, node_positions = _load_graph_and_positions(circuits_dir)

    node_csv = results_dir / "node_data.csv"
    if not node_csv.exists():
        typer.echo(f"[ERROR] {node_csv} not found — run optimize.py first.")
        raise typer.Exit(1)

    learned: dict[int, float] = {}
    test_prob: dict[int, float] = {}
    adversary: dict[int, float] = {}
    with node_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            node = int(row["node"])
            learned[node] = float(row["learned_rate"])
            test_prob[node] = float(row["test_prob"])
            adversary[node] = float(row["adversary"])

    # ── Figure 1: the game ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _draw_panel(
        axes[0], graph, node_positions, learned,
        cmap=plt.colormaps["jet"], vmax=max(learned.values(), default=1.0) or 1.0,
        title="Learned noise (observed trap-failure rate)", cbar_label="failure rate",
    )
    _draw_panel(
        axes[1], graph, node_positions, test_prob,
        cmap=plt.colormaps["viridis"], vmax=max(test_prob.values(), default=1.0) or 1.0,
        title="Optimal trap distribution (per-node test probability)", cbar_label="test prob",
    )
    _draw_panel(
        axes[2], graph, node_positions, adversary,
        cmap=plt.colormaps["magma"], vmax=max(adversary.values(), default=1.0) or 1.0,
        title="Adversary (LP dual: hardest errors)", cbar_label="attack weight",
    )
    fig.suptitle(
        f"Trap-distribution optimisation — Problem 1 on the learned heatmap  (n={n_qubits}, d={depth})",
        fontsize=13,
    )
    out_game.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_game, bbox_inches="tight")
    typer.echo(f"Saved → {out_game}")

    # ── Figure 2: detection-rate comparison ─────────────────────────────────
    rates_json = results_dir / "detection_rates.json"
    if rates_json.exists():
        with rates_json.open() as fh:
            detection_rates = json.load(fh)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        names = list(detection_rates.keys())
        values = [detection_rates[n] for n in names]
        colors = ["tab:green" if n == "OptimizedTraps" else "tab:gray" for n in names]
        bars = ax2.bar(names, values, color=colors)
        ax2.set_ylabel("worst-case detection rate")
        ax2.set_ylim(0, 1)
        ax2.set_title("Detection rate: optimised vs. baselines (learned error set)")
        for bar, v in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        fig2.savefig(out_rates, bbox_inches="tight")
        typer.echo(f"Saved → {out_rates}")


if __name__ == "__main__":
    app()
