# The FK12 analogue for Clifford + MSI circuits (bipartite traps)

This documents the verification scheme implemented in [`bro_circuit.py`](bro_circuit.py). It
is the circuit-model analogue of **FK12** (the bipartite, two-colour trap protocol used for
MBQC brickwork in `applications/benchmark-stim`). The goal: verify a Clifford+MSI computation
with **exactly two test-run types** and **no graph-colouring search**, by compiling the
circuit à la Broadbent so its trap graph is provably bipartite.

It is meant to be reused — `fk12_bro_test_runs(circuit, n_qubits)` returns the two test runs
for any Broadbent-skeleton circuit.

---

## 1. Setting: traps for Clifford + MSI

From the magic-blindness paper (arXiv:2601.07111, §4 *Verification*), in a test (magic-free)
round the whole computation is a single Clifford `G` on `N = width + ancillas` wires, all
measured in the computational basis. A **trap** for wire `q` is an input state stabilised by

```
    stab_q = G† Z_q G          (a single signed Pauli string over N wires).
```

If the input is a `+1` eigenstate of `stab_q`, then `G|ψ⟩` is a `±1` eigenstate of `Z_q`, so
the outcome of wire `q` is **deterministic** in an honest run. A Pauli deviation `E` is
**detected** by trap `q` iff `{Z_q, E} = 0`, i.e. `E` has an `X` or `Y` on wire `q`. Trapping a
set that covers all *useful* wires therefore catches every *harmful* deviation (one that flips
a useful outcome) — this is the paper's Lemma 5.

A trap can be checked on its own test run, but that is wasteful. Two traps `q, q'` can be
**merged** into one test run iff they share a product `+1` eigenstate, which holds iff their
stabilisers **commute on every wire index**:

> **Compatible** `stab_q, stab_q'` ⟺ on each wire `k`, the single-qubit letters are equal or
> at least one is `I` (they never disagree as two different non-identity Paulis).

Build the **incompatibility graph** `Inc`: one node per trap, an edge for each *incompatible*
pair. Merging traps into the fewest test runs = **colouring `Inc`**; one colour class = one
test run (a set of mutually compatible traps). In general this colouring is NP-hard, and the
number of test runs (the chromatic number) depends on the circuit `G` — both undesirable.

## 2. Broadbent compilation makes `Inc` bipartite (χ = 2)

Broadbent (2018, *How to Verify a Quantum Computation*) compiles any circuit so that the
**Clifford part uses only `H` and `CNOT`**, pushing all non-Clifford content into magic-state
injection:

```
    H = H T T H T T H T T H          P = T T          (each T = one MSI)
```

The Clifford skeleton of an `H` is a 6-ancilla gadget (`H` + `CNOT` only); each `T` is an MSI
gadget `F = SWAP ∘ CNOT` (magic-free, hence pure Clifford, in a test round). So a
Broadbent-compiled test-round circuit is **entirely `H` + `CNOT`**.

**Claim (paper §4, stated without proof there).** For an `H + CNOT` circuit the trap
incompatibility graph is **bipartite with chromatic number 2** — recovering Broadbent's two
test runs (his "X-test" and "Z-test").

**Mechanism (why it must be 2).** `H` is the *only* gate that exchanges `X ↔ Z`; `CNOT`
*preserves Pauli type per wire* (`Z_c→Z_c, Z_t→Z_c Z_t`; `X_c→X_c X_t, X_t→X_t`). So in an
`H + CNOT` circuit the Pauli *type* that each trap places on each wire is fixed by the **parity
of Hadamards** crossing that wire — there are exactly two type-assignments, hence two
compatibility classes, hence `Inc` is bipartite. The two colour classes are precisely
Broadbent's two global test states. (Conversely `CZ` *mixes*, `X_a → X_a Z_b`; with a `CZ`
gate set and `X`-basis traps the graph can have odd cycles and is **not** bipartite — that is
the only regime where the construction fails.)

**Verification.** Reproduced over 1500 random circuits in
`applications/quasi_graph/broadbent_skeleton_bipartite.ipynb` (non-bipartite count = 0), and
again here with the F/MSI gadget added (`bro_circuit.py` test runs are always exactly two).
That notebook also documents a *confounded* earlier experiment (CZ + X-basis) that wrongly
looked non-bipartite — the cautionary tale behind insisting on `CNOT + H`, Z-basis.

## 2a. Reading the 2-colouring straight off the circuit (no graph, O(gates))

We do **not** build `Inc` at all. The colouring is structural: *a Hadamard flips a wire's
colour*, and a two-qubit gate couples the two wires. Concretely (`segment_two_colouring`):

* Rewrite `G` over `{H, CZ}` (`CNOT c t = H_t · CZ_{c,t} · H_t`; `SWAP = 3 CNOT`).
* Cut each wire's timeline into **segments** at every `H`; join consecutive segments (they
  take **opposite** colours — the flip). Each `CZ` joins the two wires' *current* segments.
* The resulting **segment graph** (paths + edges) is bipartite; BFS 2-colours it. Wire `q`'s
  colour is its **last** segment's colour.

This is the quasi-graph of `applications/reviews/notes_verification.md`, and it is **verified
to be exactly a proper colouring of `Inc`** (0 monochromatic edges over random circuits;
matching `two_colouring` and identical end-to-end `p_failed_round`). Cost is **O(gates)** — it
touches neither the tableau nor an adjacency matrix.

> **Pitfall.** The tempting shortcut "colour = (#H on physical wire `q`) mod 2" is **wrong**:
> H-parity colours each gadget correctly but only up to an *independent per-gadget flip*, and
> the data `CNOT`s then make those flips globally inconsistent (≈1100 monochromatic edges over
> 60 random circuits). The segment graph is precisely what carries the inter-gadget coupling
> the bare per-wire count drops. This mirrors the `benchmark-stim` TODO: the generic FK12 path
> was cubic (`build_stabilizer`) until a structural closed form replaced it — same move here.

The same segment sweep also yields each wire's **first**-segment colour (`segment_colours`
returns `(col_first, col_last)`). That is all `fk12_bro_test_runs` needs — **no tableau, no
stabilisers** — because of an invariant verified over 11 400+ stabilisers (§3): every canonical
trap of a Broadbent skeleton is **sign-`+`** and **`Y`-free**. So the whole setup is **O(gates)**,
matching the brickwork's closed form — the `O(N²)` tableau is gone entirely.

## 3. The protocol — Broadbent's X-test / Z-test in closed form (O(gates))

This is exactly Broadbent's two test runs (arXiv:1509.09180, lines 292–293): the **X-test runs
the identity on `|0⟩^n`**, the **Z-test on `|+⟩^n`**, and *"performing an `H` locally swaps the
X- and Z-test runs"* (his line 720) — the segment-colour flip. Generalised to our random
skeletons, with the colouring `(col_first, col_last)`:

For **test run `C ∈ {0,1}`**:
1. **Trap wires** `= {q : col_last[q] = C}`, each with **expected outcome `0`**.
2. **Input prep:** wire `k` is fed `|+⟩` (an `X`/`Z(π)` eigenstate) iff `col_first[k] ≠ C`,
   else `|0⟩` (a `Z` eigenstate). Broadbent's `|0⟩→Z`, `|+⟩→X` feed rule, with the wire's basis
   read off its *input* segment colour.
3. **Check.** Prepare the product state, apply `G`, measure all wires; the round **fails** iff
   any trap wire's outcome is `1`.

This needs no per-trap stabiliser or sign because of the invariant: every `stab_q = G† Z_q G`
is sign-`+` (so expected `= 0`) and `Y`-free (so the prep is purely `|0⟩`/`|+⟩`, never `|+i⟩`).
The invariant is a property of real Cliffords `⟨H, CNOT⟩` with Z-input traps; it is *verified*
in the test suite against the tableau oracle (`canonical_z_traps` + `two_colouring`):
noiseless-deterministic and identical `p_failed_round`. Wires the merge leaves unconstrained
get an arbitrary basis — harmless, since trap failure under (depolarising) noise is
input-independent.

`fk12_bro_test_runs` returns the two `BroTestRun`s (`prep_letters`, `traps`, `expected`). In
the full protocol the Client interleaves computation rounds with test rounds, each test round
picking one of the two runs uniformly, and aborts if more than `w` of `s` test rounds fail
(identical accept/abort logic to FK12 / the paper's Protocol). The bipartite structure fixes
the **detection rate at 1/2** independently of `G` — the property RandomTraps only achieves in
expectation and which a general-circuit colouring would make `G`-dependent.

## 4. Relationship to the other benchmarks

| | `benchmark-stim` (MBQC) | `benchmark-stim-msi` | `benchmark-stim-msi-bro` (this) |
|---|---|---|---|
| circuit | brickwork **graph state** | random Clifford + MSI | **Broadbent skeleton** + MSI |
| params | width × depth | n × t | **width × depth** (like MBQC) |
| traps | FK12 bipartite (2 colours) | RandomTraps (random subset) | **FK12 analogue** (2 colours) |
| detection rate | 1/2 (bipartite) | 1/2 (in expectation) | **1/2 (bipartite, `G`-independent)** |
| test runs / cell | 2 | many (sampled) | **2** |
| colouring | free, structural (brickwork bipartition) | n/a | free, structural (`segment_colours`, O(gates)) |
| setup cost | O(V) (closed form) | O(N²) per trap pool | **O(gates)** (closed form: colouring + `|0⟩`/`|+⟩` prep; no tableau) |

The win over `benchmark-stim-msi`'s RandomTraps: only **two** compiled circuits per cell (so
sampling is cheap and shots are effectively free again, as in FK12), a fixed `1/2` detection
rate that does not depend on the circuit, and no per-round subset sampling. The cost is the
Broadbent compilation overhead (6 ancillas per `H`-gadget), i.e. the MBQC-like growth in wire
count — which is exactly why the parameters are `(width, depth)`.

**Caveat.** Traps here are taken on *all* `N` wires (a superset of the paper's useful set
`Q = {output} ∪ {MSI ancillas}`). Bipartite on all wires ⟹ bipartite on `Q`, so this is the
stronger/secure choice; restricting to `Q` would match the paper's protocol exactly and make
the colouring cheaper, at the cost of a more intricate "useful-wire" bookkeeping.
