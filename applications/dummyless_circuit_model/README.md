# Dummyless stabilizer search — circuit model (Broadbent compilation)

Searching for **dummyless** stabilizer test runs in the **circuit model**, using the
Broadbent-compiled Clifford skeleton (the `H = HTTHTTHTTH` H-gadget), rather than in a
graph state. Background: `applications/reviews/notes_verification.md`,
§*Pauli noise detection in Clifford circuits via stabilizer testing*.

## The problem

For a Clifford `C` on `N` wires, the **canonical stabilizer basis** is

```
S_i = C† P_i C ,   i = 0 … N-1
```

where `P` is the output measurement basis. A stabilizer is **dummyless** if its Pauli
string has **no Z** (only I, X, Y) — then any +1-eigenstate is a product of *XY-plane*
single-qubit states, so no `|0>` "dummy" qubits are needed.

Test runs compose: the XOR of measurement outcomes ↔ the **product** of stabilizers. So
we may change basis by multiplying canonical stabilizers. The quest:

> From `{S_0, …, S_{N-1}}`, build as many **linearly independent dummyless** stabilizers
> as possible. For graph states (CZ-only Cliffords) `n-1` is always reachable in
> polynomial time (notes §*The case of a Graph*). Is it reachable in the circuit model?

**Caveat that makes this combinatorial, not linear:** "no Z, but Y allowed" is *not* a
linear subspace of the symplectic space, because `X · Y = iZ` at a shared index. So you
cannot just solve a linear system — you must search over products of generators.

## Result on the minimal example

One Broadbent H-gadget = **7 wires** (data wire 0 + 6 ancillas; the role exits on wire 6).
In Broadbent's **Z-basis** (`S_i = C† Z_i C`) the search finds **6 = n−1** independent
dummyless stabilizers — the target. (The MBQC **X-basis** convention only reaches rank 3
on this circuit; the CNOT skeleton is "graph-like" in the Z-basis, not the X-basis.)

```
S_0 = +XZ_____      none of the canonical S_i is dummyless
S_1 = +_ZZ____      (each carries a Z)
S_2 = +ZXXZ___
S_3 = +___ZZ__
S_4 = +__ZXXZ_
S_5 = +_____ZZ
S_6 = +____ZXX

6 independent dummyless products:
  +YYX__XX   = S0·S2·S3·S6
  +X__XYYX   = S0·S1·S4·S6
  +YYYYYYX   = S0·S2·S4·S6
  +YXY__XX   = S0·S1·S2·S3·S6
  +X__YXYX   = S0·S1·S3·S4·S6
  -YYX__YY   = S0·S2·S3·S5·S6
```

## Files

| file | what it is |
|---|---|
| `dummyless.py`       | the library: circuit builder, canonical basis, `Stab`, `Workspace` |
| `minimal_example.py` | guided walkthrough on the single H-gadget (run this first) |
| `bigger_example.py`  | bigger circuits + the graph-based construction, in phases |
| `mbqc_view.py`       | the measurement-based view: resource graph + the `F_A` invariant |
| `playground.py`      | a template you edit to try your own circuits / combinations |
| `NOTES.md`           | the MBQC reframing, the natural invariant, and why it resolves the rest |

Run with the project venv:

```bash
./.venv/bin/python applications/dummyless_circuit_model/minimal_example.py
./.venv/bin/python applications/dummyless_circuit_model/playground.py
```

## Manipulating stabilizers yourself

```python
from dummyless import Workspace

ws = Workspace.single_hadamard(basis="Z")   # 7-wire Broadbent H-gadget

ws.show_basis()                  # print S_0 … S_6
ws.combine(0, 1)                 # product S_0 · S_1  (try s_1 s_2 etc.)
ws.combine(0, 2, 3, 6)           # product S_0 · S_2 · S_3 · S_6
ws.s[0] * ws.s[2] * ws.s[6]      # same idea, operator style

st = ws.combine(0, 2, 3, 6)
st.is_dummyless                  # True / False
st.pauli                         # the stim.PauliString
st.factors                       # [0, 2, 3, 6] — which generators it's built from
ws.eigenstate_str(st)            # the XY-plane input state it asks for (dummyless only)

ws.check_set([(0,1,4,6), (0,2,3,6), (0,2,4,6)])   # rank + dummyless count of a set
ws.search_dummyless()            # auto-find a maximal independent dummyless set
```

### Choosing a different circuit

```python
# two H-gadgets in series on one role (13 wires):
ws = Workspace.from_logical(1, [("H", 0), ("H", 0)], basis="Z")

# two roles, CNOT between them, then an H-gadget on role 0:
ws = Workspace.from_logical(2, [("E", 0, 1), ("H", 0)], basis="Z")

# X-basis (MBQC notes convention) instead of Z-basis (Broadbent):
ws = Workspace.single_hadamard(basis="X")

# CZ-based "quasi-graph" gadget instead of the CNOT skeleton:
ws = Workspace.single_hadamard(style="quasi_graph")
```

Logical ops are on **roles**: `("E", r, s)` for an entangling gate (CNOT under
`style="broadbent"`, CZ under `style="quasi_graph"`) and `("H", r)` for an H-gadget.

## Is the search graph-based? (and the incompatibility-graph idea)

`search_dummyless` / `all_dummyless` are **brute force** — enumerate all `2^N` products,
keep the dummyless ones, greedily extract an independent set. That is *not* the polynomial
graph algorithm of `veriphix.protocols.Dummyless`.

But the graph algorithm's **structure transfers**, which `Workspace.constructive_dummyless`
makes explicit in phases (run `bigger_example.py`):

* **Rfull = product of all canonical generators is dummyless** — the exact starting point
  of the notes' algorithm (`Rfull` has X/Y at every qubit, never Z, whenever each qubit has
  an odd number of X-owners; true for H-gadget chains).
* **"even-degree removals" → "removable singles".** `R\i = Rfull · Sᵢ` stays dummyless iff
  `Sᵢ` **agrees with Rfull** (same X vs Y) at every qubit it owns. Pure-Z generators are
  always removable. In the pure-graph case this predicate *is* "deg(i) even"; here it is the
  right local generalisation.
* **"odd-degree nodes" → the "disagree set"** = the X-owners of Rfull's `Y` positions. It is
  even in size and must be **paired up**. The pairing is driven by the **role trajectories**
  (each H-gadget is a short 1-D chain whose two ends are a disagree pair connected through
  the gadget's own ancillas) plus **direct cross-gadget pairs** for CNOT coupling. This is
  the circuit analogue of `odd_pair_generators_bfs`'s spanning tree of odd nodes.

Phase results (`constructive_dummyless`):

| circuit | A: Rfull+removables | B: +trajectory pairing (poly) | C: +brute residual | n−1 |
|---|---|---|---|---|
| one H-gadget (N=7)            | **6** | 6 | 6 | 6 |
| two H-gadgets (N=13)          | **12**| 12| 12| 12|
| 3 roles, CNOT **chain** (N=15)| 10 | **14** | 14 | 14 |
| 4 roles, CNOT **ring** (N=22) | 3  | 13 | **21** | 21 |

So the polynomial Phase A+B reaches `n−1` for **H-gadget chains and CNOT-chain circuits**
(the trajectory pairing was needed for N=15). The boundary is **whether Rfull is dummyless**:
a CNOT **ring** makes a qubit acquire an *even* number of X-owners, so Rfull itself picks up
a Z, the whole Rfull-anchored construction collapses (A reaches only 3/21), and only brute
force (C) reaches `n−1`. That residual — and even some Rfull-dummyless cases the trajectory
pairing misses — is exactly the open question in the notes
(*"is dummyless feasible for all Cliffords?"*). Over 120 random circuits the pairing reaches
`n−1` in ~58% of Rfull-dummyless cases vs ~27% otherwise: Rfull-dummyless is **necessary but
not sufficient**.

**Why not just feed the incompatibility graph to `protocols.Dummyless`?** Because that
function builds *graph-state* stabilizers `S_v = X_v ∏_{w~v} Z_w` and keys everything off
graph **degree**. Our circuit stabilizers are not of that form, and the incompatibility
graph's degree **mispredicts** removability (for the H-gadget the even-degree node is the
one removal that *fails*, while the odd-degree nodes succeed). Its *paths* are still useful
for the pairing, but the even/odd classification must come from the agreement-with-Rfull
predicate, not from the incompatibility graph. `Workspace.incompatibility_graph()`,
`.rfull()`, `.removable_singles()` are exposed so you can explore this yourself.

## Notes on scaling

`all_dummyless()` / `search_dummyless()` enumerate all `2^N` products — fine for the MWE
and up to ~20 wires, exponential beyond. `constructive_dummyless()` does the polynomial
phases first and only falls back to brute force for the residual. Finding the maximum-rank
dummyless set for *arbitrary* circuits in polynomial time is the open research question.
