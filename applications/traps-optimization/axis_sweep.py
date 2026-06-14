"""axis_sweep.py — biased (single-axis) vs two-axis harmful noise, done right.

Blindness confines trap *measurements* to the X-Y plane (the random +θ padding
is undone by a pre-Z(θ) rotation, which only commutes through CZ in-plane).
*Dummies* — Z-eigenstate preparations — are allowed, and they matter: measuring
a node's neighbour places a Z on that node through the CZ entanglement (the node
acts as a dummy), and that Z anticommutes with BOTH X and Y there.

So the correct per-node detection conditions for a generator-subset trap
(measured set T, graph adjacency Γ) are:
    detect X_v  ⟺  (Γc)_v = 1            (odd measured neighbours of v)
    detect Y_v  ⟺  c_v ⊕ (Γc)_v = 1
    detect Z_v  ⟺  c_v = 1               (Z is harmless — omitted from ℰ)
A single trap detects BOTH X_v and Y_v iff c_v = 0 and (Γc)_v = 1, i.e. v is a
dummy with an odd number of measured neighbours (S_v = Z).

Consequences (verified):
  * single-axis harmful noise (X-only or Y-only) → detection rate 1.0;
  * two-axis harmful noise (X and Y, e.g. depolarising) → NOT capped at 1/2:
    the dummy mechanism lifts it, up to 1.0 when a GF(2) "Z-cover" of the noisy
    region exists. The achievable rate is a graph/region property.
The 1/2 only appears if you refuse the dummy mechanism (measure noisy nodes
directly only) — which is why the pool here includes the noisy nodes' neighbours.

Output: results/axis_sweep.json + plots/axis_sweep.pdf

Usage
-----
    python applications/traps-optimization/axis_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import typer
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12, OptimizedTraps, get_bipartite_coloring, get_node_positions
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.trap_optimization import multi_basis_independent_set_pool
from veriphix.verifying import TrappifiedSchemeParameters

from pauli_errors import pauli_errors

app = typer.Typer(add_completion=False)
SAMPLED_BASE = Path("applications/gospel/sampled_circuits")

# harmful-noise axis structures (Z is harmless and omitted)
AXIS_SETS = {"single-axis (X)": "X", "single-axis (Y)": "Y", "two-axis (X,Y)": "XY"}


def _rate(graph, noisy, paulis):
    # Pool must include the noisy nodes' neighbours so the dummy mechanism is
    # available (measuring a neighbour places Z on a noisy node → catches X & Y).
    region = set(noisy)
    for v in list(noisy):
        region |= set(graph.neighbors(v))
    protocol = OptimizedTraps(
        errors=pauli_errors(graph, noisy, paulis),
        test_pool=lambda g: multi_basis_independent_set_pool(g, restrict_to=region),  # X,Y only
    )
    protocol.create_test_runs(graph)
    return round(protocol.detection_rate, 4)


def _is_independent(graph, noisy):
    nl = list(noisy)
    return all(not graph.has_edge(u, v) for i, u in enumerate(nl) for v in nl[i + 1:])


def _row(name, graph, noisy):
    return {
        "config": name,
        "n_noisy": len(noisy),
        "independent": _is_independent(graph, noisy),
        "rates": {label: _rate(graph, noisy, ax) for label, ax in AXIS_SETS.items()},
    }


def _load_brickwork():
    with sorted((SAMPLED_BASE / "circuits-3-5-1e-1").glob("*.qasm"))[0].open() as f:
        circuit = read_qasm(f)
    pattern = transpile(circuit)
    pattern.minimize_space()
    pos = {n: (int(p[0]), int(p[1])) for n, p in get_node_positions(pattern).items()}
    red, blue = get_bipartite_coloring(pattern)
    client = Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=FK12(manual_colouring=(red, blue)),
        parameters=TrappifiedSchemeParameters(comp_rounds=0, test_rounds=1, threshold=0),
    )
    return client.graph, pos, sorted(red)


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("applications/traps-optimization/results"),
    out_plot: Annotated[Path, typer.Option(help="Output PDF")] = Path("applications/traps-optimization/plots/axis_sweep.pdf"),
) -> None:
    rows = []
    # canonical graphs
    rows.append(_row("path9: independent {0,2,4,6,8}", nx.path_graph(9), {0, 2, 4, 6, 8}))
    rows.append(_row("path9: connected {0,1,2,3}", nx.path_graph(9), {0, 1, 2, 3}))
    rows.append(_row("C5: all (odd cycle)", nx.cycle_graph(5), set(range(5))))
    rows.append(_row("star6: leaves (independent)", nx.star_graph(6), {1, 2, 3, 4, 5, 6}))

    # real brickwork: independent colour class vs connected blob
    graph, pos, red = _load_brickwork()
    by_col: dict[int, list[int]] = {}
    for node, (col, _r) in pos.items():
        by_col.setdefault(col, []).append(node)
    connected_region = {n for c in range(8, 13) for n in by_col.get(c, [])}
    rows.append(_row("brickwork: independent (colour class)", graph, set(red[:15])))
    rows.append(_row("brickwork: connected blob", graph, connected_region))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "axis_sweep.json").write_text(json.dumps({"ceiling_two_axis": 0.5, "rows": rows}, indent=2))

    # plot
    labels = list(AXIS_SETS.keys())
    colors = ["#2c7fb8", "#7fcdbb", "#d95f0e"]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6))
    for k, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (k - 1) * width, [r["rates"][label] for r in rows], width, label=label, color=color)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="naive ceiling 1/2 (no dummies)")
    ax.axhline(1.0, color="green", ls=":", lw=1, label="single-axis ceiling = 1.0")
    ax.set_ylabel("optimal worst-case detection rate (X/Y traps + dummies)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Harmful-noise detection (blind MBQC): single-axis → 1.0; two-axis lifted above 1/2 by the dummy mechanism")
    ax.set_xticks(x)
    xt = ax.set_xticklabels([r["config"] for r in rows], rotation=25, ha="right", fontsize=8)
    for tick, r in zip(xt, rows):
        if r["independent"]:
            tick.set_color("green"); tick.set_fontweight("bold")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight")

    typer.echo(f"\n{'config':40s} {'indep':6s} {'X':>6s} {'Y':>6s} {'X,Y':>6s}")
    for r in rows:
        rr = r["rates"]
        typer.echo(f"{r['config']:40s} {str(r['independent']):6s} "
                   f"{rr['single-axis (X)']:6.3f} {rr['single-axis (Y)']:6.3f} {rr['two-axis (X,Y)']:6.3f}")
    typer.echo(f"\nSaved → {out_plot}")


if __name__ == "__main__":
    app()
