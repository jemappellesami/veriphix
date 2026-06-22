"""Playground -- edit this file to manipulate stabilizers yourself.

Run:  ./.venv/bin/python applications/dummyless_circuit_model/playground.py

Everything you need lives on the `Workspace` object `ws`:

    ws.s[i]                 the canonical stabilizer S_i (a `Stab`)
    ws.s[i] * ws.s[j]       their product (XORs supports, multiplies Paulis)
    ws.combine(i, j, k)     product S_i · S_j · S_k  (same thing, list-style)
    st.is_dummyless         True if the Pauli string has no Z
    st.pauli                the underlying stim.PauliString
    st.factors              which canonical generators it is built from
    ws.eigenstate_str(st)   the XY-plane product input state (dummyless only)
    ws.check_set([...])     rank / dummyless count of a proposed generating set
    ws.search_dummyless()   auto-find a maximal independent dummyless set

Change the circuit at the top, then edit the EXPERIMENT block.
"""

from dummyless import Workspace

# --------------------------------------------------------------------------- #
#  CHOOSE YOUR CIRCUIT                                                         #
# --------------------------------------------------------------------------- #
# Minimal working example: one H-gadget, 7 wires.
ws = Workspace.single_hadamard(basis="Z", style="broadbent")

# Other examples to try (uncomment one):
#
#   # two H-gadgets in series on the same role (13 wires):
#   ws = Workspace.from_logical(1, [("H", 0), ("H", 0)], basis="Z")
#
#   # two roles, a CNOT between them, then an H-gadget on role 0:
#   ws = Workspace.from_logical(2, [("E", 0, 1), ("H", 0)], basis="Z")
#
#   # the MBQC / X-basis convention instead of Broadbent's Z-basis:
#   ws = Workspace.single_hadamard(basis="X")
#
#   # the CZ-based ("quasi-graph") gadget instead of the CNOT skeleton:
#   ws = Workspace.single_hadamard(basis="Z", style="quasi_graph")

ws.show_basis()
print()

# --------------------------------------------------------------------------- #
#  EXPERIMENT -- edit freely                                                   #
# --------------------------------------------------------------------------- #

# Try a single product by hand:
print("S_0 · S_1        :", ws.combine(0, 1))
print("S_0 · S_1 · S_2  :", ws.combine(0, 1, 2))
print("operator style   :", ws.s[0] * ws.s[2] * ws.s[3] * ws.s[6])
print()

# Propose your own generating set and check it (list of generator-index tuples):
my_set = [
    (0, 1, 4, 6),
    (0, 2, 3, 6),
    (0, 2, 4, 6),
]
report = ws.check_set(my_set)
print(f"my_set: {report['dummyless']}/{report['count']} dummyless, "
      f"rank {report['rank']}, "
      f"{'all independent' if report['independent'] else 'has dependencies'}")
print()

# Or just let the search find a maximal independent dummyless set:
print("Auto-search:")
ws.search_dummyless()
