# Dummyless Trap Generation

---

## Part I — For Humans

### Context: trap-based verification

In trap-based verification a Client delegates a quantum computation G to an untrusted Server. The key result (Lemma 3 of the paper) is that any malicious Server behaviour reduces, after a Pauli twirl, to a **Pauli deviation** E applied at the output of G just before measurement. Pauli deviations split into two classes:

- **Harmful**: E has X or Y on at least one *useful* output qubit (output wire or MSI ancilla wire). These can flip a measurement outcome and corrupt the computation.
- **Harmless**: E is purely Z on every useful qubit. Z commutes with computational-basis measurement and leaves outcomes unchanged.

The goal is to design **traps** — classically simulable test computations that detect every harmful deviation.

### What is a trap?

A generalised trap for a subset Q of useful qubits works as follows:

1. Prepare an input state that is a +1 eigenstate of `Ŝ_Q = G† (∏_{q∈Q} Z_q) G`.
2. Run the computation G on this input.
3. Check that the **parity** of the output bits `{q ∈ Q}` equals 0.

If a deviation E is present, the parity is flipped iff E **anticommutes** with `∏_{q∈Q} Z_q`, i.e. E has X or Y on an **odd number of qubits in Q**. This is the detection condition:

    Trap Q detects E  ⟺  {∏_{q∈Q} Z_q ,  E} = 0.

### What is a dummyless trap?

The input state must be a **product state** (tensor product of single-qubit states) — otherwise the Client needs extra ancilla qubits called *dummies* to prepare it. Preparing a product eigenstate of `Ŝ_Q = G†(∏_{q∈Q} Z_q)G` is possible without dummies **iff `Ŝ_Q` has no bare Z**: every qubit that carries Z in `Ŝ_Q` must also carry X (i.e. be Y, not pure Z). This is the **dummyless condition**:

    supp(Z-part of Ŝ_Q) ⊆ supp(X-part of Ŝ_Q).

### The goal: detect all harmful deviations with dummyless traps

For verification to work, every harmful deviation must be detected by **at least one** dummyless trap. A sufficient condition (Lemma 4 of the paper): for each useful qubit q there exists a dummyless trap Q with q ∈ Q. Then a deviation with X or Y on qubit q is caught by that trap.

The **ideal** is a single-qubit trap `Q = {q}` for each useful qubit q — if each `G†Z_qG` is dummyless, one trap per qubit suffices and complete detection is guaranteed. If `G†Z_qG` has bare Z for some useful qubit q, a **generalised trap** (product over multiple qubits) must be found that is dummyless and still covers q.

### Solved case: graph-state Cliffords

The two frameworks differ sharply for graph states `G = (∏_{(i,j)∈E} CZ_{ij}) · H^⊗N`:

**Circuit model** (Z-basis measurements, back-propagate Z):

    G† Z_v G = X_v    for every qubit v.

Every back-propagated Z operator is pure X — **trivially dummyless**. Single-qubit traps work for every qubit immediately with no construction needed.

**MBQC** (X-basis measurements, back-propagate X):

    G† X_v G = Z_v · ∏_{u∈N(v)} X_u    for every qubit v.

Every back-propagated X operator has a **bare Z at qubit v** — the single-qubit trap is not dummyless. Finding dummyless products is non-trivial. The equivalent formulation uses the graph state stabilizers `S_v = G Z_v G† = X_v Z_{N(v)}` (forward conjugation); dummyless condition becomes supp(Γc) ⊆ supp(c) (closed neighbourhood on the graph). A poly-time construction gives N−1 independent dummyless traps in O(N):

- **Rfull** = ∏_v S_v: X everywhere (dummyless ✓)
- **R\v** for each even-degree v: Rfull · S_v, X everywhere except v (dummyless ✓)
- **R\(u,w)** for each odd-degree pair: product along a BFS path (dummyless ✓)

The one remaining undetectable harmless direction is E* = Z on all odd-degree nodes (a stabilizer-group element that commutes with all X-basis measurements).

### The CNOT + H-gadget circuits

The circuits of interest are built from n data qubits, H-gadgets (each using 6 ancilla qubits a1..a6), and CNOT gates between data qubits. With t H-gadgets the circuit acts on N = n + 6t qubits.

For these circuits, each back-propagated stabilizer `G†Z_iG` carries **both X and Z components** — none of the N single-qubit back-propagated operators is trivially dummyless. Products of them can cancel bare Z's, but finding dummyless products efficiently is the open problem.

### The open problem

> Find, in **polynomial time in N**, a set of dummyless products `{∏_{q∈Q} G†Z_qG}` such that every useful output qubit q belongs to at least one trap Q in the set.

- The brute-force search over all 2^N subsets Q is O(2^N · N) — exponential.
- No poly-time construction is known for H-gadget circuits.
- For the graph-state case the poly-time construction exploits A = I (pure-X stabilizers); this special structure is absent here.

---

## Part II — For LLMs

### Setup

Let G be a Clifford unitary on N qubits. The *back-propagated stabilizers* are

    s_i = G† Z_i G,   i = 0, …, N−1.

Encode them as two binary matrices A, B ∈ GF(2)^{N×N}:

    A[i,j] = 1  iff  s_i  has X or Y on qubit j
    B[i,j] = 1  iff  s_i  has Z or Y on qubit j

A coefficient vector c ∈ GF(2)^N selects a product stabilizer with:

    X-part = Aᵀc mod 2,    Z-part = Bᵀc mod 2.

The corresponding trap set is `Q = supp(c)` (qubits where c[i] = 1).

### Dummyless condition

The product stabilizer `∏_{i: c[i]=1} G†Z_iG` is dummyless iff:

    supp(Bᵀc) ⊆ supp(Aᵀc)     (no bare Z in the product Pauli string)

### Detection condition

In the circuit-model framework a deviation E is detected by trap c iff:

    {∏_{i: c[i]=1} Z_i ,  E} = 0

i.e. E has X or Y on an **odd number** of qubits in `supp(c)`.  
For a **binary error pattern** e_X (e_X[j] = 1 iff E has X or Y on qubit j), this is:

    c · e_X = 1   mod 2.

**Important:** the detection is by `c` itself (the trap indicator), NOT by `Aᵀc` (the X-part of the input stabilizer). These coincide only when A = I (graph states).

### Complete detection

Let Q_useful ⊆ {0,...,N-1} be the set of useful output qubits. Complete detection requires: for every non-empty S ⊆ Q_useful, some dummyless c with `c · 1_S = 1 mod 2`. Equivalently: the restriction of the dummyless c-vectors to useful-qubit coordinates must **span GF(2)^{|Q_useful|}**.

Minimally: for each useful qubit q, there is a dummyless c with c[q] = 1.

### Solved case: graph-state Cliffords (A = I, B = Γ)

For G = (∏ CZ) · H^⊗N:

- G† Z_i G = X_i  ⟹  A = I, B = Γ (adjacency matrix).
- Every c trivially satisfies dummyless when B = 0 (pure-X generators, i.e. isolated nodes); in general the condition reduces to a closed-neighbourhood condition on the graph.
- Detection: c · e_X = Aᵀc · e_X = c · e_X (since A = I). So X-part of stabilizer = indicator of Q.
- Poly-time construction (O(N)) gives N−1 independent dummyless traps covering all N qubits; the one undetectable harmless direction is E* = Z on all odd-degree nodes.

### CNOT + H-gadget circuits

**Circuit.** H-gadget on qubit d (ancillas a1..a6):

    H(d)
    CNOT(a1→d), CNOT(a2→a1), H(a2)
    CNOT(a3→a2), CNOT(a4→a3), H(a4)
    CNOT(a5→a4), CNOT(a6→a5), H(a6)

Plus CNOT gates between data qubits. Total: N = n + 6t qubits.

**Observed structure.**

- Every G†Z_iG has both X and Z components (all back-propagated stabilizers have bare Z).
- rank(A) < N: some qubits never appear in the X-part of any single G†Z_iG.
- Products ∏_{q∈Q} G†Z_qG can cancel bare Z's — dummyless products exist and can be found by brute force.
- Brute-force for N=14: 2^14 candidates, ~640 dummyless products found, detection rank 7/14 (over c-vectors restricted to useful qubits — this number depends on which qubits are deemed useful).

**The open problem (precise formulation).**

Given A, B ∈ GF(2)^{N×N} from an H-gadget Clifford circuit, find in **poly(N)** time a set of vectors {c₁, ..., c_r} ⊆ GF(2)^N such that:

1. Each c_k is dummyless: supp(Bᵀc_k) ⊆ supp(Aᵀc_k)
2. For every useful qubit q ∈ Q_useful, some c_k has c_k[q] = 1
3. The restriction of {c_k} to Q_useful spans GF(2)^{|Q_useful|}

**Note:** condition 3 implies complete detection of all harmful deviations. Condition 1 guarantees no dummy qubits are needed.

### Key contrast with graph states

| | Graph-state G | H-gadget G |
|---|---|---|
| G†Z_iG | pure X (A=I, B=Γ) | X + Z mixed |
| Single-qubit traps | all dummyless | none dummyless |
| Detection vector = c? | yes (A=I ⟹ Aᵀc=c) | no (A≠I) |
| Poly-time construction | O(N) via spanning tree | unknown |
| Harmless undetectable | E* (one direction) | unknown |

### Hint from graph case

For graph states, the poly-time construction exploits:
- Rfull (c=1...1): always dummyless, covers all qubits
- Removing one generator at a time while preserving dummylessness
- Graph structure (odd/even degree) guides which removals are safe

For H-gadget circuits, the analogous approach would need to:
- Identify a starting dummyless product (possibly a large product that cancels all bare Z's)
- Systematically modify it to generate independent traps covering all useful qubits
- Do this without checking all 2^N possibilities
