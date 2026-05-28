"""plot.py — aggregate per-circuit CSVs and render the failure-rate heatmap.

Usage
-----
    python applications/gospel/hotgate/plot.py
    python applications/gospel/hotgate/plot.py \\
        --results-dir applications/gospel/hotgate/results \\
        --out applications/gospel/hotgate/heatmap.pdf \\
        --circuits-dir applications/gospel/circuits/circuits-3-6 \\
        --p-ent 2e-3 --n-test-rounds 100
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
from matplotlib.colors import Normalize
from typing_extensions import Annotated

import typer

app = typer.Typer(add_completion=False)


def _load_graph_and_positions(circuits_dir: Path):
    """Load the brickwork graph and node grid positions from the first circuit."""
    from veriphix.protocols import get_node_positions
    from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
    from veriphix.sampling_circuits.qasm_parser import read_qasm
    from veriphix.blinding import Secrets
    from veriphix.client import Client
    from veriphix.protocols import FK12, get_bipartite_coloring
    from veriphix.verifying import TrappifiedSchemeParameters

    qasm_files = sorted(circuits_dir.glob("*.qasm"))
    if not qasm_files:
        raise RuntimeError(f"No .qasm files found in {circuits_dir}")

    with qasm_files[0].open() as f:
        from veriphix.sampling_circuits.qasm_parser import read_qasm
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


SAMPLED_BASE = Path("applications/gospel/sampled_circuits")


@app.command()
def main(
    n_qubits:      Annotated[int,   typer.Option(help="Number of qubits")]                           = 3,
    depth:         Annotated[int,   typer.Option(help="Circuit depth")]                              = 6,
    bqp_error:     Annotated[str,   typer.Option(help="BQP error tag (folder suffix, e.g. 1e-1)")]  = "1e-1",
    results_dir:   Annotated[Path,  typer.Option(help="Directory with per-circuit CSV files")]       = Path("applications/gospel/hotgate/results"),
    out:           Annotated[Path,  typer.Option(help="Output PDF path")]                            = Path("applications/gospel/hotgate/heatmap.pdf"),
    p_ent:         Annotated[float, typer.Option(help="Noise level label for the plot title")]       = 2e-3,
    n_test_rounds: Annotated[int,   typer.Option(help="Test rounds label for the plot title")]       = 100,
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"
    csv_files = sorted(results_dir.glob("circuit_*.csv"))
    if not csv_files:
        typer.echo(f"[ERROR] No circuit_*.csv files found in {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Aggregating {len(csv_files)} circuit CSVs …")

    failure_counts: defaultdict[int, int] = defaultdict(int)
    total_tests:    defaultdict[int, int] = defaultdict(int)
    # node_positions read from CSVs directly — no need to reload patterns.
    node_positions: dict[int, tuple[int, int]] = {}

    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                node = int(row["node"])
                node_positions[node] = (int(row["col"]), int(row["row"]))
                failure_counts[node] += int(row["failure_count"])
                total_tests[node]    += int(row["total_tests"])

    n_circuits = len(csv_files)

    # Build 2-D rate grid.
    n_cols = max(c for c, _ in node_positions.values()) + 1
    n_rows = max(r for _, r in node_positions.values()) + 1

    rate_grid = np.zeros((n_rows, n_cols))
    for node, (col, row) in node_positions.items():
        n = total_tests[node]
        rate_grid[row, col] = failure_counts[node] / n if n > 0 else 0.0

    typer.echo(f"  grid: {n_cols} cols × {n_rows} rows  |  "
               f"max failure rate: {rate_grid.max():.3f}")

    # Load graph edges for drawing.
    typer.echo("Loading graph edges …")
    graph, _ = _load_graph_and_positions(circuits_dir)

    # Plot.
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.6), max(5, n_rows * 1.2)))

    cmap   = plt.colormaps["jet"]
    norm   = Normalize(vmin=0.0, vmax=1.0)
    radius = 0.38

    for u, v in graph.edges():
        cu, ru = node_positions[u]
        cv, rv = node_positions[v]
        ax.plot([cu, cv], [ru, rv], "k-", linewidth=0.6, zorder=1)

    for node, (col, row) in node_positions.items():
        color  = cmap(norm(rate_grid[row, col]))
        circle = mpatches.Circle((col, row), radius, color=color, zorder=2)
        ax.add_patch(circle)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    n_qubits = len(set(r for _, r in node_positions.values()))
    n_depth  = n_cols
    ax.set_title(
        f"FK12 trap failure rate  "
        f"(n={n_qubits}, d={n_depth}, {n_circuits} circuits × {n_test_rounds} rounds, "
        f"$p_{{\\mathrm{{ent}}}}$={p_ent:.0e})",
        pad=10,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    typer.echo(f"Saved → {out}")


if __name__ == "__main__":
    app()
