"""The measurement-based view of the Broadbent compilation, and its natural invariant.

Run:  ./.venv/bin/python applications/dummyless_circuit_model/mbqc_view.py

Idea (the key reframing): the Broadbent compilation is *measurement-based*. Every gadget
is magic-state injection (MSI), and MSI of a T = measuring the data qubit in the
`|±_{π/4}⟩` basis -- an **XY-plane measurement**. So the whole compiled circuit is an MBQC
pattern on a **resource graph**, with XY-plane measurements (Clifford skeleton = angle-0/X
measurements in test runs) and a computational-(Z)-basis readout on the BQP output.

In this view the dummyless theory of arXiv:2303.08865 applies *verbatim* to the compiled
**resource graph** `G`:

  * graph-state stabilisers  `S_v = X_v · ∏_{w~v} Z_w`,
  * `R_full = ∏_v S_v` is dummyless (X/Y everywhere; Y exactly on odd-degree nodes),
  * there are always `|V|-1` independent dummyless generators,
  * the unique undetected direction is the **F_A invariant**

        U = ∏_{v : deg(v) odd} Z_v     (Z on every odd-degree node of G),

    which is *harmless* (Lemma "harmless" of the paper): reflection through the X-Y plane
    leaves every XY-plane measurement projector invariant, so applying U before the
    measurements does not change the output distribution.

Why this matters here: the messy behaviour we saw in the *circuit/tableau* representation
(CNOT skeleton, Z-basis canonical `C†Z_iC`) -- the Pauli-rank vs detection-rank gap, and the
CNOT *ring* making `R_full` non-dummyless -- were **artefacts of that representation**. The
resource-graph (MBQC) view is clean for *every* circuit, ring included, because the compiled
object is always a graph state and the paper's graph construction is unconditional.

The one genuinely circuit-model wrinkle is the **output qubit**: it is read in the Z basis
(the BQP bit), not the XY plane, so `F_A` does not govern it. But `U` puts a `Z` on the
output node, and `Z` commutes with a Z-basis readout -- so that part of the invariant is
harmless for the *opposite* reason (readout, not `F_A`). The invariant therefore splits:

    U  =  (Z on internal odd-degree nodes : harmless by F_A)
        · (Z on output odd-degree nodes   : harmless because outputs are Z-measured).

Only depends on stim + networkx.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import stim

from dummyless import gf2_rank


# --------------------------------------------------------------------------- #
#  Resource graph of a Broadbent-compiled circuit (the MBQC view)              #
# --------------------------------------------------------------------------- #

def resource_graph(num_roles: int, logical_ops: list[tuple]) -> tuple[nx.Graph, dict[int, int]]:
    """The MBQC resource graph of the compiled circuit.

    Each H-gadget on a role hangs a 6-node PATH off the role's current node (the CZ chain
    of the gadget; the gadget's local Hadamards only rotate measurement bases, not the
    graph). Each entangling gate is a CZ edge between the two roles' current nodes.

    Returns (graph, holder) where holder[r] is role r's final (output) node.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_roles))
    holder = {r: r for r in range(num_roles)}
    nxt = num_roles
    for op in logical_ops:
        if op[0] in ("E", "CZ"):
            g.add_edge(holder[op[1]], holder[op[2]])
        elif op[0] == "H":
            prev = holder[op[1]]
            for _ in range(6):
                g.add_node(nxt)
                g.add_edge(prev, nxt)
                prev = nxt
                nxt += 1
            holder[op[1]] = prev
        else:
            raise ValueError(f"unknown op {op!r}")
    return g, holder


def graph_stabiliser(g: nx.Graph, v: int, n: int) -> stim.PauliString:
    """S_v = X_v · ∏_{w~v} Z_w."""
    xs = np.zeros(n, bool)
    zs = np.zeros(n, bool)
    xs[v] = True
    for w in g.neighbors(v):
        zs[w] = True
    return stim.PauliString.from_numpy(xs=xs, zs=zs)


def z_on(nodes, n: int) -> stim.PauliString:
    zs = np.zeros(n, bool)
    for v in nodes:
        zs[v] = True
    return stim.PauliString.from_numpy(xs=np.zeros(n, bool), zs=zs)


def is_dummyless(p: stim.PauliString) -> bool:
    return all(p[k] != 3 for k in range(len(p)))


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #

def analyse(name: str, num_roles: int, logical_ops: list[tuple]) -> None:
    g, holder = resource_graph(num_roles, logical_ops)
    n = g.number_of_nodes()
    S = [graph_stabiliser(g, v, n) for v in range(n)]
    odd = [v for v in g.nodes() if g.degree(v) % 2 == 1]
    outputs = set(holder.values())
    U = z_on(odd, n)

    r_full = stim.PauliString(n)
    for s in S:
        r_full = r_full * s

    # maximal independent set of dummyless generators
    basis: list[int] = []
    tests: list[stim.PauliString] = []
    for size in range(1, n + 1):
        for combo in itertools.combinations(range(n), size):
            p = stim.PauliString(n)
            for i in combo:
                p = p * S[i]
            if not is_dummyless(p):
                continue
            v = sum(1 << i for i in combo)
            x = v
            for b in basis:
                x = min(x, x ^ b)
            if x:
                basis.append(x)
                basis.sort(reverse=True)
                tests.append(p)

    supports = [sum(1 << k for k in range(n) if t[k] in (1, 2)) for t in tests]
    det_rank = gf2_rank(supports)

    print(f"== {name} ==")
    print(f"   resource graph: |V| = {n}, output nodes = {sorted(outputs)}, "
          f"odd-degree nodes = {odd}")
    print(f"   R_full = {r_full}   (X/Y everywhere; Y on odd-degree)")
    print(f"   independent dummyless generators = {len(tests)}  (|V|-1 = {n - 1})")
    print(f"   detection rank = {det_rank}  ->  undetected dim = {n - det_rank}")
    print(f"   F_A invariant  U = {U}")
    print(f"     commutes with every dummyless test : {all(U.commutes(t) for t in tests)}")
    print(f"     odd-degree OUTPUT nodes (Z-readout-harmless): {sorted(set(odd) & outputs)}")
    print(f"     odd-degree INTERNAL nodes (F_A-harmless)    : {sorted(set(odd) - outputs)}")
    print()


def main() -> None:
    print("Measurement-based view: the natural invariant of Clifford + MSI is")
    print("U = Z on odd-degree nodes of the compiled resource graph (the F_A reflection).\n")
    analyse("single H-gadget", 1, [("H", 0)])
    analyse("two H-gadgets in series", 1, [("H", 0), ("H", 0)])
    analyse("3 roles, CZ chain + 2 H-gadgets", 3,
            [("E", 0, 1), ("E", 1, 2), ("H", 0), ("H", 1)])
    analyse("4 roles, CZ RING + 3 H-gadgets", 4,
            [("E", 0, 1), ("E", 1, 2), ("E", 2, 3), ("E", 3, 0),
             ("H", 0), ("H", 2), ("H", 3)])


if __name__ == "__main__":
    main()
