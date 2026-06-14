"""pentagon_experiment.py — the fractional-colouring win, on an odd-cycle graph.

Uses the pentagon-with-tails resource graph from `pentagon_circuit.py` (a real
5-qubit CZ-pentagon + single-qubit-gate circuit, transpiled to MBQC). Because
the graph has an odd cycle (girth 5, non-bipartite), the standard-trap detection
rate optimum is 1/χ_f(C5) = 2/5, while a naive proper colouring (FK12 greedy)
gives only 1/χ(C5) = 1/3. The LP recovers the fractional optimum.

Contrast: on the bipartite brickwork χ_f = χ = 2, so greedy already reaches the
1/2 optimum and the LP shows no gap. The odd cycle is what exposes the win.

Output: results/pentagon_rates.json + plots/pentagon_rates.pdf

Usage
-----
    python applications/traps-optimization/pentagon_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim
import typer
from typing_extensions import Annotated

from veriphix.protocols import FK12, OptimizedTraps
from veriphix.trap_optimization import independent_set_pool

from pentagon_circuit import make_client, pentagon_nodes, verify_pentagon_graph

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("applications/traps-optimization/results"),
    out_plot: Annotated[Path, typer.Option(help="Output PDF")] = Path("applications/traps-optimization/plots/pentagon_rates.pdf"),
) -> None:
    client = make_client()
    graph = client.graph
    summary = verify_pentagon_graph(graph)  # asserts odd-cycle structure
    typer.echo(f"Graph verified: {summary['n_nodes']} nodes, girth {summary['girth']}, "
               f"bipartite={summary['bipartite']}")

    nodes = list(graph.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    pent = set(pentagon_nodes(graph))

    def z(v: int) -> stim.PauliString:
        s = ["I"] * len(nodes)
        s[index[v]] = "Z"
        return stim.PauliString("".join(s))

    errors = [z(v) for v in pent]  # detect Z on the odd-cycle nodes

    # standard-trap pools, restricted to the pentagon (the noisy region)
    optimized = OptimizedTraps(errors=errors, test_pool=lambda g: independent_set_pool(g, restrict_to=pent))
    optimized.create_test_runs(graph)
    fk_greedy = FK12()
    fk_greedy.create_test_runs(graph)

    rates = {
        "OptimizedTraps (fractional)": round(optimized.detection_rate, 4),
        "FK12 greedy (integer colouring)": round(fk_greedy.detection_rate, 4),
    }
    refs = {"1/χ_f(C5) = 2/5": 2 / 5, "1/χ(C5) = 1/3": 1 / 3, "bipartite ceiling 1/2": 0.5}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pentagon_rates.json").write_text(json.dumps({"rates": rates, "refs": refs}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(rates.keys())
    vals = [rates[n] for n in names]
    bars = ax.bar(names, vals, color=["tab:green", "tab:gray"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    ax.axhline(2 / 5, color="green", ls=":", lw=1.2, label="1/χ_f(C5) = 2/5 (fractional optimum)")
    ax.axhline(1 / 3, color="red", ls="--", lw=1, label="1/χ(C5) = 1/3 (integer colouring)")
    ax.axhline(0.5, color="k", ls=":", lw=0.8, label="bipartite ceiling = 1/2")
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("worst-case detection rate (standard traps)")
    ax.set_title("Odd-cycle (pentagon) graph: LP fractional colouring beats integer colouring")
    ax.legend(fontsize=8, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=8)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight")

    typer.echo("Detection rates: " + ", ".join(f"{k}={v}" for k, v in rates.items()))
    typer.echo(f"Saved → {out_plot}")


if __name__ == "__main__":
    app()
