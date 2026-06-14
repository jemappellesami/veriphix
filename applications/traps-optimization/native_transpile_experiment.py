"""native_transpile_experiment.py — native (non-brickwork) transpilation of the
sampled QASM circuits, and why it does NOT expose the LP advantage.

Hypothesis (to test): transpiling the .qasm circuits with graphix's *native*
transpiler (instead of the brickwork transpiler) gives non-bipartite resource
graphs, so the LP optimisation would beat a proper colouring.

Finding (this script): it does **not**. Every sampled circuit is built from
CNOT + single-qubit rotations, and graphix's native CNOT gadget inserts ancilla
nodes — so a CNOT-based cycle becomes an *even* graph cycle. The resource graphs
are therefore **bipartite** (often trees), χ_f = χ = 2, and the LP only *matches*
a proper 2-colouring (both at 1/χ_f). Only direct CZ gates create odd cycles
(see pentagon_experiment.py); CNOT circuits cannot.

For each circuit we report: bipartite?, girth, and the standard-trap detection
rate from (a) the LP and (b) FK12 greedy — which coincide on bipartite graphs.

Output: results/native_transpile.json

Usage
-----
    python applications/traps-optimization/native_transpile_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import stim
import typer
from typing_extensions import Annotated

from veriphix.protocols import FK12, OptimizedTraps
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.trap_optimization import independent_set_pool

app = typer.Typer(add_completion=False)
SAMPLED_BASE = Path("applications/gospel/sampled_circuits")


def _z_errors(graph: nx.Graph) -> list[stim.PauliString]:
    nodes = list(graph.nodes)
    out = []
    for i in range(len(nodes)):
        s = ["I"] * len(nodes)
        s[i] = "Z"
        out.append(stim.PauliString("".join(s)))
    return out


@app.command()
def main(
    n_qubits: Annotated[int, typer.Option(help="Circuit width")] = 3,
    depth: Annotated[int, typer.Option(help="Circuit depth")] = 5,
    bqp_error: Annotated[str, typer.Option(help="BQP tag")] = "1e-1",
    n_circuits: Annotated[int, typer.Option(help="How many circuits to inspect")] = 12,
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("applications/traps-optimization/results"),
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"
    files = sorted(circuits_dir.glob("*.qasm"))[:n_circuits]

    rows = []
    for f in files:
        with f.open() as fh:
            circuit = read_qasm(fh)
        # NATIVE transpiler (graphix), not brickwork
        graph = circuit.transpile().pattern.extract_graph()
        bipartite = nx.is_bipartite(graph)
        girth = min((len(c) for c in nx.minimum_cycle_basis(graph)), default=None)

        errors = _z_errors(graph)
        opt = OptimizedTraps(errors=errors, test_pool=independent_set_pool)
        opt.create_test_runs(graph)
        fk = FK12()
        fk.create_test_runs(graph)

        rows.append({
            "circuit": f.name,
            "n_nodes": graph.number_of_nodes(),
            "bipartite": bipartite,
            "girth": girth,
            "lp_rate": round(opt.detection_rate, 4),
            "greedy_rate": round(fk.detection_rate, 4),
        })

    n_nonbip = sum(not r["bipartite"] for r in rows)
    n_gap = sum(r["lp_rate"] > r["greedy_rate"] + 1e-9 for r in rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "native_transpile.json").write_text(
        json.dumps({"n_circuits": len(rows), "n_non_bipartite": n_nonbip,
                    "n_with_lp_gap": n_gap, "rows": rows}, indent=2)
    )

    typer.echo(f"\n{'circuit':18s} {'nodes':>5s} {'bip':>5s} {'girth':>6s} {'LP':>6s} {'greedy':>7s}")
    for r in rows:
        typer.echo(f"{r['circuit']:18s} {r['n_nodes']:5d} {str(r['bipartite']):>5s} "
                   f"{str(r['girth']):>6s} {r['lp_rate']:6.3f} {r['greedy_rate']:7.3f}")
    typer.echo(f"\nnon-bipartite: {n_nonbip}/{len(rows)}   circuits where LP beats greedy: {n_gap}/{len(rows)}")
    typer.echo("=> CNOT+rotation circuits transpile to bipartite graphs; LP only matches greedy.")
    typer.echo("   Need direct CZ gates for odd cycles (see pentagon_experiment.py).")


if __name__ == "__main__":
    app()
