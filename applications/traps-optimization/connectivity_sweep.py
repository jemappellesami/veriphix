"""connectivity_sweep.py — detection rate vs. how connected the noisy region is.

The structure sweep showed a sharp split (independent region → beats 1/2;
connected region → capped).  Here we sweep *continuously* between the two by
growing the noisy region as a graph ball of radius ``r`` around a centre node
(the same notion of "region" used by the Gaussian-region noise model): r=0 is a
single node, larger r pulls in adjacent nodes and raises the internal edge count.

For each radius we report the full-depolarising ({X,Y,Z}) optimal detection rate
over the basis-diverse pool, and the number of internal edges of the region — so
the x-axis is a genuine "how independent is my noise" knob.  The rate falls from
2/3 (sparse / independent) toward 1/χ_f (dense) as connectivity rises.

Output: results/connectivity_sweep.json  +  plots/connectivity_sweep.pdf

Usage
-----
    python applications/traps-optimization/connectivity_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import stim
import typer
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12, get_bipartite_coloring, get_node_positions
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.trap_optimization import (
    build_detection_matrix,
    multi_basis_independent_set_pool,
    solve_trap_distribution,
)
from veriphix.verifying import TrappifiedSchemeParameters

app = typer.Typer(add_completion=False)
SAMPLED_BASE = Path("applications/gospel/sampled_circuits")


def _load_brickwork_graph():
    with sorted((SAMPLED_BASE / "circuits-3-5-1e-1").glob("*.qasm"))[0].open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    pos = {node: (int(p[0]), int(p[1])) for node, p in get_node_positions(pattern).items()}
    red, blue = get_bipartite_coloring(pattern)
    client = Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=FK12(manual_colouring=(red, blue)),
        parameters=TrappifiedSchemeParameters(comp_rounds=0, test_rounds=1, threshold=0),
    )
    return client.graph, pos


def _two_axis_errors(graph, noisy):
    """Harmful two-axis deviations {X_v, Y_v} (Z is harmless, omitted)."""
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    out = []
    for v in sorted(noisy):
        for p in "XY":
            s = ["I"] * len(nodes)
            s[index[v]] = p
            out.append(stim.PauliString("".join(s)))
    return out


@app.command()
def main(
    center:   Annotated[int,  typer.Option(help="Centre node of the growing region")] = 31,
    max_radius: Annotated[int, typer.Option(help="Maximum BFS radius")]                = 6,
    out_dir:  Annotated[Path, typer.Option(help="Output directory")]                   = Path("applications/traps-optimization/results"),
    out_plot: Annotated[Path, typer.Option(help="Output PDF")]                         = Path("applications/traps-optimization/plots/connectivity_sweep.pdf"),
) -> None:
    graph, pos = _load_brickwork_graph()

    radii, rates, edges, sizes = [], [], [], []
    for r in range(max_radius + 1):
        ball = set(nx.single_source_shortest_path_length(graph, center, cutoff=r))
        n_edges = graph.subgraph(ball).number_of_edges()
        # include neighbours so the dummy mechanism is available
        region = set(ball)
        for v in list(ball):
            region |= set(graph.neighbors(v))
        pool = multi_basis_independent_set_pool(graph, restrict_to=region)  # X,Y only
        matrix = build_detection_matrix(graph, pool, _two_axis_errors(graph, ball))
        rate = round(solve_trap_distribution(matrix).detection_rate, 4)
        radii.append(r); rates.append(rate); edges.append(n_edges); sizes.append(len(ball))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "connectivity_sweep.json").write_text(
        json.dumps({"center": center, "radii": radii, "rates": rates,
                    "internal_edges": edges, "region_sizes": sizes}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(edges, rates, "o-", color="tab:red")
    for e, rt, sz in zip(edges, rates, sizes):
        ax.annotate(f"{sz}n", (e, rt), textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="naive ceiling 1/2 (no dummies)")
    ax.set_xlabel("internal edges of noisy region (connectivity →)")
    ax.set_ylabel("two-axis {X,Y} optimal detection rate")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"Two-axis harmful-noise detection vs. region connectivity (brickwork, centre {center})\n"
                 "labels = region size in nodes; pool uses dummies (neighbour measurements)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight")

    typer.echo("radius  size  edges  rate")
    for r, sz, e, rt in zip(radii, sizes, edges, rates):
        typer.echo(f"  {r:>2}    {sz:>3}   {e:>4}   {rt:.4f}")
    typer.echo(f"Saved → {out_plot}")


if __name__ == "__main__":
    app()
