"""optimize.py — client-side trap-distribution optimisation (Problem 1).

Pipeline (no quantum simulation — everything is computed from the learned
heatmap and the graph):

  1. Load the n=3, d=5 brickwork graph + node positions.
  2. Read the *learned* trap-failure heatmap (observed rates) from the
     noise-learning results — what the client measured about its own device.
  3. Turn the noisy nodes into a learned error set ℰ = {Z_v : v noisy}.
  4. Solve Problem 1 with OptimizedTraps (independent sets of the noisy
     subgraph) → optimal test distribution, detection rate ε, dual adversary.
  5. Compare ε against the baselines FK12 (greedy + bipartite) and RandomTraps.
  6. Persist everything for plotting.

Outputs (in ``results/``):
  * ``node_data.csv``    : node, col, row, learned_rate, test_prob, adversary
  * ``detection_rates.json``

Usage
-----
    python applications/traps-optimization/optimize.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import typer
from typing_extensions import Annotated

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import (
    FK12,
    OptimizedTraps,
    RandomTraps,
    get_bipartite_coloring,
    get_node_positions,
)
from veriphix.sampling_circuits.brickwork_state_transpiler import transpile
from veriphix.sampling_circuits.qasm_parser import read_qasm
from veriphix.trap_optimization import independent_set_pool
from veriphix.verifying import TrappifiedSchemeParameters

from errors import load_learned_heatmap, noisy_nodes, z_errors

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
    return client.graph, node_positions, (red, blue)


@app.command()
def main(
    n_qubits:    Annotated[int,   typer.Option(help="Number of qubits")]                             = 3,
    depth:       Annotated[int,   typer.Option(help="Circuit depth")]                                = 5,
    bqp_error:   Annotated[str,   typer.Option(help="BQP error tag")]                                = "1e-1",
    heatmap_dir: Annotated[Path,  typer.Option(help="Learned-heatmap results dir")]                  = Path("applications/noise_learning/results"),
    threshold:   Annotated[float, typer.Option(help="Failure-rate threshold for a node to be 'noisy'")] = 0.05,
    out_dir:     Annotated[Path,  typer.Option(help="Output directory")]                             = Path("applications/traps-optimization/results"),
) -> None:
    circuits_dir = SAMPLED_BASE / f"circuits-{n_qubits}-{depth}-{bqp_error}"
    graph, node_positions, (red, blue) = _load_graph_and_positions(circuits_dir)
    nodes = list(graph.nodes)
    pos_to_node = {pos: node for node, pos in node_positions.items()}

    # 1–3. learned heatmap → error set
    rates = load_learned_heatmap(heatmap_dir, pos_to_node)
    noisy = noisy_nodes(rates, threshold)
    errors = z_errors(graph, noisy)
    typer.echo(f"Learned heatmap: {len(noisy)} noisy nodes (threshold {threshold}) → |ℰ|={len(errors)}")
    if not errors:
        typer.echo("[ERROR] no noisy nodes found — lower the threshold or run noise_learning first.")
        raise typer.Exit(1)

    # 4. optimise
    optimized = OptimizedTraps(
        errors=errors,
        test_pool=lambda g: independent_set_pool(g, restrict_to=noisy),
    )
    pool = optimized.create_test_runs(graph)
    distribution = optimized.distribution
    assert distribution is not None
    typer.echo(f"OptimizedTraps: |ℋ|={len(pool)} independent sets, detection rate = {optimized.detection_rate:.4f}")

    # per-node test probability (coverage) = Σ p(canvas) over canvases containing the node
    test_prob: dict[int, float] = {node: 0.0 for node in nodes}
    for p_i, run in zip(distribution, pool, strict=True):
        for trap in run.traps:
            (node,) = trap
            test_prob[node] += float(p_i)

    # per-node adversary weight (dual), mapped back from error index → node
    adversary_by_node: dict[int, float] = {node: 0.0 for node in nodes}
    for err_pauli, weight in zip(errors, optimized.adversary, strict=True):
        (idx,) = err_pauli.pauli_indices("Z")
        adversary_by_node[nodes[idx]] = float(weight)

    # 5. baselines
    fk_greedy = FK12()
    fk_greedy.create_test_runs(graph)
    fk_bipartite = FK12(manual_colouring=(red, blue))
    fk_bipartite.create_test_runs(graph)
    detection_rates = {
        "OptimizedTraps": optimized.detection_rate,
        "FK12_greedy": fk_greedy.detection_rate,
        "FK12_bipartite": fk_bipartite.detection_rate,
        "RandomTraps": RandomTraps().detection_rate,
    }
    typer.echo("Detection rates: " + ", ".join(f"{k}={v:.4f}" for k, v in detection_rates.items()))

    # 6. persist
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "node_data.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node", "col", "row", "learned_rate", "test_prob", "adversary"])
        for node in nodes:
            col, row = node_positions[node]
            writer.writerow([
                node, col, row,
                rates.get(node, 0.0),
                test_prob[node],
                adversary_by_node[node],
            ])
    with (out_dir / "detection_rates.json").open("w") as fh:
        json.dump(detection_rates, fh, indent=2)

    typer.echo(f"Saved → {out_dir}/node_data.csv, detection_rates.json")


if __name__ == "__main__":
    app()
