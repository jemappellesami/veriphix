"""Build Pauli-error sets ℰ over a noisy region, with a tunable number of types.

The ceiling experiment varies how many single-qubit Pauli types the noise spans:
  * 1 type  (e.g. {Z})       — pure dephasing
  * 2 types (e.g. {Z, X})    — biased
  * 3 types ({X, Y, Z})      — full depolarising
For each, ℰ = {P_v : v ∈ noisy, P ∈ types}, as Pauli strings in
``list(graph.nodes)`` order (matching ``build_stabilizer``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

if TYPE_CHECKING:
    import networkx as nx


def pauli_errors(graph: nx.Graph, noisy: set[int], paulis: str) -> list[stim.PauliString]:
    """Return ``{P_v : v ∈ noisy, P ∈ paulis}`` as Pauli strings."""
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    errors: list[stim.PauliString] = []
    for v in sorted(noisy):
        for p in paulis:
            s = ["I"] * n
            s[index[v]] = p
            errors.append(stim.PauliString("".join(s)))
    return errors
