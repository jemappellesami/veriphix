"""pentagon_circuit.py — a 5-qubit circuit whose MBQC resource graph is an
odd cycle (pentagon) with tails.

Why: the brickwork graphs used elsewhere are *bipartite*, so the standard-trap
detection rate is pinned at 1/χ_f = 1/2 and the LP optimisation can never beat
it. To expose the fractional-chromatic gap (where the LP genuinely wins) we need
a graph with an **odd cycle**. The 5-cycle C5 has χ_f = 5/2, so the optimal
standard-trap detection rate on the pentagon nodes is 1/χ_f = 2/5 — strictly
below 1/2, and strictly *above* the 1/3 of a naive proper 3-colouring (FK12
greedy). That gap is the whole point.

Construction:
  * 5 qubits, CZ between them in a pentagon (cz(i, i+1 mod 5)) — applied FIRST,
    so they connect the input nodes 0..4 into a clean 5-cycle;
  * then single-qubit rotations on each qubit, which the MBQC transpiler turns
    into a chain of measured nodes hanging off each pentagon node (a "tail").

Resulting resource graph: pentagon on nodes {0,1,2,3,4} + one tail per node.

Run this file to build, transpile, and verify the graph is as predicted:
    python applications/traps-optimization/pentagon_circuit.py
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from graphix import Circuit

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import FK12
from veriphix.verifying import TrappifiedSchemeParameters

N_QUBITS = 5


def build_pentagon_circuit(gates_per_qubit: int = 2, seed: int = 1234) -> Circuit:
    """Build the 5-qubit pentagon-CZ + single-qubit-gate circuit.

    The pentagon CZ gates are applied before any single-qubit gate, so they
    entangle the input nodes directly (giving the clean C5); the single-qubit
    rotations then extend a tail off each pentagon node.
    """
    rng = np.random.default_rng(seed)
    circuit = Circuit(N_QUBITS)

    # pentagon of CZ gates on the input wires
    for i in range(N_QUBITS):
        circuit.cz(i, (i + 1) % N_QUBITS)

    # single-qubit gates -> tails (continuations) on each qubit
    for q in range(N_QUBITS):
        for _ in range(gates_per_qubit):
            circuit.rz(q, float(rng.uniform(0, 2 * np.pi)))
            circuit.rx(q, float(rng.uniform(0, 2 * np.pi)))

    return circuit


def pentagon_pattern(gates_per_qubit: int = 2, seed: int = 1234):
    """Transpile the pentagon circuit to an MBQC pattern (resource graph kept clean)."""
    return build_pentagon_circuit(gates_per_qubit, seed).transpile().pattern


def make_client(gates_per_qubit: int = 2, seed: int = 1234) -> Client:
    """A veriphix Client on the pentagon-with-tails pattern."""
    pattern = pentagon_pattern(gates_per_qubit, seed)
    return Client(
        pattern=pattern,
        secrets=Secrets(r=True, a=True, theta=True),
        protocol=FK12(),
        parameters=TrappifiedSchemeParameters(comp_rounds=0, test_rounds=1, threshold=0),
    )


def pentagon_nodes(graph: nx.Graph) -> list[int]:
    """The 5 input nodes that form the odd cycle (they are 0..4 by construction)."""
    return list(range(N_QUBITS))


def verify_pentagon_graph(graph: nx.Graph) -> dict:
    """Check the resource graph is a pentagon (odd 5-cycle) with tails.

    Returns a summary dict; raises AssertionError if the structure is wrong.
    """
    pent = pentagon_nodes(graph)
    # 1. the five input nodes form a 5-cycle
    cycle_edges = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]
    for u, v in cycle_edges:
        assert graph.has_edge(u, v), f"missing pentagon edge ({u},{v})"
    # 2. the pentagon is a genuine odd cycle -> graph is NOT bipartite
    assert not nx.is_bipartite(graph), "graph is bipartite (no odd cycle!)"
    # 3. shortest cycle is the pentagon (girth 5)
    girth = min(len(c) for c in nx.minimum_cycle_basis(graph))
    assert girth == 5, f"expected girth 5, got {girth}"
    # 4. each pentagon node carries a tail (degree 3: two cycle neighbours + tail)
    for v in pent:
        assert graph.degree(v) == 3, f"pentagon node {v} has degree {graph.degree(v)} (expected 3)"
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "pentagon_nodes": pent,
        "bipartite": nx.is_bipartite(graph),
        "girth": girth,
        "chi_f_C5": "5/2 -> standard-trap ceiling 1/chi_f = 2/5",
    }


if __name__ == "__main__":
    client = make_client()
    graph = client.graph
    summary = verify_pentagon_graph(graph)
    print("Pentagon resource graph verified ✓")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  pentagon edges: {[(i,(i+1)%N_QUBITS) for i in range(N_QUBITS)]}")
    print(f"  tails (degree-1 leaves): {[n for n in graph.nodes if graph.degree(n)==1]}")
