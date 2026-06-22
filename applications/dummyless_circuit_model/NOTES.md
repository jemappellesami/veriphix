# Notes: the measurement-based view and the natural invariant

## The reframing

The Broadbent compilation is **measurement-based**: every gadget is magic-state injection
(MSI), and MSI of a `T` measures the data qubit in the `|±_{π/4}⟩` basis — an **XY-plane
measurement** (see SO26 §MSI). So the whole compiled circuit is an **MBQC pattern on a
resource graph**: XY-plane measurements (the Clifford skeleton uses angle-0/X measurements
in test runs), with a computational-(Z)-basis readout on the BQP output qubit.

## The natural invariant (answer to "is there an invariant for Clifford + MSI?")

Yes — it is exactly the `F_A` reflection of arXiv:2303.08865, transferred to the compiled
**resource graph** `G`:

```
U = ∏_{v : deg(v) odd} Z_v        (Z on every odd-degree node of G)
```

`U` is the reflection through the X–Y plane realised as a physical unitary. Because every
internal node is measured in the XY plane (MSI), reflecting through that plane leaves the
measurement projectors invariant (Lemma "harmless"), so applying `U` before the measurements
does not change the output — it is harmless.

Verified by `mbqc_view.py` on chains, a CNOT/CZ chain, and a CZ ring:

| circuit | \|V\| | dummyless gens | detection rank | undetected dim | `U` commutes w/ all tests |
|---|---|---|---|---|---|
| one H-gadget       | 7  | 6  | 6  | 1 | ✓ |
| two H-gadgets      | 13 | 12 | 12 | 1 | ✓ |
| 3 roles, CZ chain  | 15 | 14 | 14 | 1 | ✓ |
| 4 roles, CZ **ring** | 22 | 21 | 21 | 1 | ✓ |

So in the graph view there are always `|V|-1` independent dummyless generators (the paper's
graph theorem, **unconditional**), detection is full, and `U` is the **unique undetected
direction**.

## Why this resolves the earlier mess

Working in the *circuit/tableau* representation (CNOT skeleton, Z-basis canonical
`S_i = C†Z_iC`) produced two confusing artefacts that are now explained away:

* **Pauli-rank ≠ detection-rank** (e.g. 6 dummyless Paulis but only rank-3 X/Y-support for
  one gadget). Artefact of the representation: in the resource-graph view, detection rank =
  `|V|-1` (full), no gap.
* **The CNOT ring broke `R_full`** (it was non-dummyless, so the Rfull-anchored construction
  collapsed). Also an artefact: in the resource-graph view the ring gives `|V|-1` dummyless
  like everything else, because the compiled object is *always* a graph state and the graph
  construction is unconditional.

The lesson: do the dummyless analysis on the **compiled resource graph with graph-state
stabilisers**, not on the unitary's tableau in a fixed basis.

## Reading the invariant off `R_full` (no graph needed)

"Odd-degree node" is only the graph-state name for a property of `R_full = ∏_i S_i` (the
product of all canonical stabilizers):

> the harmless deviation `U` = `Z` on the qubits where `R_full` carries a **`Y`**.

For a graph state `S_v = X_v ∏ Z` gives `R_full(v) = X·Z^{deg v}` = `X` (deg even) or `Y`
(deg odd), so Y-positions = odd-degree nodes — but `R_full` is defined for *any* Clifford,
so this is the graph-free handle. The deeper reading of "degree of v" is the **Z-weight
(spread) of the back-propagated observable `C†P_vC`**; only its parity matters.

**Verified** (numpy statevector, `U` read off `R_full`):

| graph | `R_full` | Y-positions = odd-degree? | X-outcome dist. invariant under U |
|---|---|---|---|
| path P₇         | `−YXXXXXY`        | ✓ {0,6}       | ✓ (≈1e-36) |
| 3-role chain    | `+XYYXXXXXYXXXXXY`| ✓ {1,2,8,14}  | ✓ (≈1e-20) |

The last column is genuine **harmlessness** (`F_A`: outcome distribution unchanged when `U`
is applied before the measurements), not just "undetected / commutes with the tests".
Holds for X (Clifford) measurements here; extends to the `π/4` magic-state angles by the
same `{I,X,Y}`-decomposition argument, so it covers the computation, not only test runs.

Caveat: needs `R_full` dummyless (X/Y everywhere) for "Y-positions" to be meaningful — the
graph-state regime. For arbitrary non-graph Cliffords the recipe still locates the undetected
direction, but harmlessness then relies on the local-Clifford-to-graph reduction.

## The one real wrinkle: the Z-basis output qubit

`F_A` governs only the XY-plane-measured (internal/MSI) nodes. The BQP output qubit is read
in the **Z** basis, so `F_A` does not cover it. But `U` places a `Z` on the output node, and
`Z` commutes with a Z-basis readout — so that part of the invariant is harmless for the
*other* reason. The invariant splits cleanly:

```
U  =  (Z on internal odd-degree nodes : harmless by F_A)
    · (Z on output   odd-degree nodes : harmless because outputs are Z-measured)
```

`mbqc_view.py` prints this split (e.g. 3-role chain: internal `{1}`, output `{2,8,14}`).

**Caveat (worth a rigorous check):** the internal-node half is the paper's lemma verbatim;
the output-node half is a natural extension argued here, not lifted directly from the paper
(whose setting is fully classical-I/O, all-XY-measured). A simulation-level check — run the
compiled pattern with and without `U` before the measurements and confirm the role-0 output
distribution is unchanged — would make it airtight. This also connects to the open BQP
question: errors on *garbage* (non-output) roles are additionally harmless (we never read
them), so the harmful set may be smaller than the single `U`, which is where a count below
`|V|-1` (toward the `N-r` idea) could come from.
