"""Bigger examples + the graph-based construction, transferred to the circuit model.

Run:  ./.venv/bin/python applications/dummyless_circuit_model/bigger_example.py

Answers three questions:

  1. Does the dummyless search keep finding n-1 on bigger circuits?  (yes, here)
  2. What is the search doing -- does it look like the graph-based approach?
     `search_dummyless` is BRUTE FORCE. But the graph algorithm's *structure*
     (`veriphix.protocols.Dummyless`: Rfull -> even-degree removals -> odd pairing)
     transfers, with "even degree" replaced by a local agreement predicate and the
     odd-pairing driven by the role trajectories. See `constructive_dummyless`:
       - Phase A (poly): Rfull + removable singles
       - Phase B (poly): trajectory-driven pairing of the disagree set
       - Phase C       : exhaustive residual (the open "all Cliffords?" part)
     Phase B now reaches n-1 on H-gadget chains AND CNOT-*chain* circuits.
  3. Can the incompatibility graph drive it? Its *paths* help, but its *degree*
     does NOT predict removability -- demonstrated at the bottom.

The CNOT *ring* shows the boundary: it makes Rfull non-dummyless, the Rfull-anchored
construction collapses (Phase A barely moves), and only brute force reaches n-1.
"""

import networkx as nx

from dummyless import Workspace, gf2_rank


EXAMPLES = [
    ("one H-gadget",            Workspace.single_hadamard(basis="Z")),
    ("two H-gadgets in series", Workspace.from_logical(1, [("H", 0), ("H", 0)], basis="Z")),
    ("3 roles, CNOT CHAIN + 2 H-gadgets  (Phase B now solves this)",
     Workspace.from_logical(3, [("E", 0, 1), ("E", 1, 2), ("H", 0), ("H", 1)], basis="Z")),
    ("4 roles, CNOT RING + 3 H-gadgets   (Rfull not dummyless -> needs brute force)",
     Workspace.from_logical(4, [("E", 0, 1), ("E", 1, 2), ("E", 2, 3), ("E", 3, 0),
                                ("H", 0), ("H", 2), ("H", 3)], basis="Z")),
]


def main() -> None:
    print("#" * 72)
    print("# 1+2.  brute-force search vs the transferred graph construction")
    print("#" * 72)
    for label, ws in EXAMPLES:
        print(f"\n=== {label}  (N = {ws.N}) ===")
        brute = ws.search_dummyless(verbose=False)
        print(f"  brute-force `search_dummyless`     : {len(brute)} independent dummyless "
              f"(target n-1 = {ws.N - 1})")
        ws.constructive_dummyless(verbose=True)

    # ------------------------------------------------------------------ #
    print("\n" + "#" * 72)
    print("# 3.  why the *incompatibility graph* is not the right graph to colour")
    print("#" * 72)
    ws = Workspace.single_hadamard(basis="Z")
    rfull = ws.rfull()
    g = ws.incompatibility_graph()
    removable, disagree = ws.removable_singles()
    print(f"\nsingle H-gadget incompatibility graph: edges = {sorted(g.edges())}")
    print(f"  degrees                 = {dict(g.degree())}")
    print(f"  even-degree nodes       = {[v for v in g.nodes if g.degree(v) % 2 == 0]}")
    print(f"  REMOVABLE (R\\i dummyless) = {removable}")
    print("  -> they do not match: the incompatibility-graph degree rule from the\n"
          "     graph-state algorithm mispredicts which removals stay dummyless,\n"
          "     because our circuit stabilizers are not of graph-state form\n"
          "     (S_i = X_i . prod Z is replaced by multi-owner / pure-Z strings).")
    print(f"\n  the right predicate is local agreement with Rfull = {rfull.pauli}:")
    for i in range(ws.N):
        owned = [k for k in range(ws.N) if ws._S[i][k] in (1, 2)]
        agree = all(ws._S[i][k] == rfull.pauli[k] for k in owned)
        print(f"    S_{i}: owns {str(owned):12}  agrees-with-Rfull={str(agree):5}  "
              f"-> {'removable' if i in removable else 'disagree'}")


if __name__ == "__main__":
    main()
