# Literature-review request: the "Clifford-skeleton via magic-state injection" compilation, and its use in error correction / detection

You are a research assistant with broad knowledge of quantum computing literature
(fault tolerance, measurement-based quantum computation, magic states, blind/verified
delegated computation). I want a careful literature review answering three questions
about a specific compilation trick. **Prioritise accuracy over coverage**: cite real
papers, flag when you are unsure, and clearly separate "established in the literature"
from "plausible but I could not find a direct reference."

## The compilation trick I'm asking about

I take a universal quantum circuit (Clifford + `T`) and rewrite it into a very regular
form built from only two ingredients:

1. **Entangling Cliffords** (`CZ`, or equivalently `CNOT`) between data qubits, and
2. **Hadamard gadgets**, where each Hadamard is realised through the identity
   `H = HPHPHPH = HTTHTTHTTH` (with `P = T²`). Each `T` is implemented by
   **magic-state injection (MSI)**: an ancilla prepared in a magic state, a Clifford
   interaction, a single-qubit measurement, and a Clifford (Pauli/`S`) byproduct
   correction.

A single `H`-gadget therefore uses **6 ancillas, 6 entangling gates (CNOT or CZ), and
4 Hadamards**, and it **teleports** the logical qubit down the ancilla chain (the data
role exits on the 6th ancilla). This is the structure used in Broadbent's
"How to Verify a Quantum Computation."

Two observations that matter for my questions:

- **The "Clifford skeleton."** If you drop the `T`-rotations and their `S`-corrections
  (which is what happens in the *test runs* of a verification protocol, where the magic
  states are replaced so the rotations become identity), you are left with a pure-Clifford
  circuit using **only `{H, CZ}` (equivalently `{H, CNOT}`) — no phase (`S`) gates.**
  That is the *real-Clifford* / "graph-state-with-Hadamards" subgroup, not the full
  Clifford group.
- **It is secretly measurement-based.** Because every `T` is an MSI = an XY-plane
  measurement at angle `π/4`, and Cliffords correspond to graph structure + Pauli-basis
  measurements, the whole compiled circuit is an **MBQC pattern on a resource graph**
  (a graph state measured in the XY plane), with a computational-(Z)-basis readout on the
  output. The resource graph is: a 6-node path per Hadamard gadget, hung off the role's
  current node, plus an edge per entangling gate.

## Why I care (context, not the question)

I'm studying *trap-based verification* of this compiled circuit — in particular
"dummyless" stabilizer tests (traps that only require XY-plane input states, no
computational-basis "dummy" qubits), and a natural **harmless deviation / invariant** of
the model: the analogue of the MBQC reflection-through-the-XY-plane symmetry, which on the
resource graph is `Z` on the odd-degree nodes (equivalently, `Z` on the `Y`-positions of
the product of all canonical stabilisers). This is the context, but the questions below
are about the **compilation trick and its appearance in error correction/detection**, not
about my verification work.

## Questions

1. **Novelty / origin.** Where does this compilation trick come from? Specifically:
   - Is the `H = HPHPHPH` / `HTTHTTHTTH` rewrite (using `P = T²` so Hadamards pass through
     the same magic-state machinery as `T`s) original to Broadbent (2018), or does it
     appear earlier / independently? Trace the lineage.
   - Is "extract the Clifford skeleton of a Clifford+T circuit and analyse its stabiliser /
     graph structure" a named, reused technique anywhere?

2. **Use in error correction and error detection.** Is this kind of structure used in QEC /
   FT? Concretely, does the literature use any of:
   - magic-state injection gadgets viewed as a **fixed resource graph / MBQC pattern**,
   - the **teleported-qubit chain** (gate teleportation) as the unit of a fault-tolerant or
     error-detecting gadget,
   - **trap / dummy qubits and stabiliser tests** as an *error-detection* primitive (not
     just for blind verification) — e.g. flag qubits, gauge fixing, detector structures in
     the language of stabiliser codes,
   - the idea that a compiled Clifford circuit has a **single harmless logical deviation**
     (a non-detectable but output-preserving error), and whether that maps onto known QEC
     notions (logical-`Z` gauge freedom, stabiliser-vs-logical distinction, transversal or
     symmetry-protected errors).

3. **Adjacent bodies of work** I should read, with the strongest 1-2 references each:
   - Measurement-based / cluster-state QEC and the graph-state ↔ stabiliser-code dictionary.
   - Magic state injection, distillation, and cultivation (fault-tolerant `T`).
   - Verified/blind delegated computation in the **prepare-and-send / circuit model**
     (Broadbent; Fitzsimons–Kashefi; Leichtle–Music–Kashefi–Ollivier; Kapourniotis et al.),
     and the **dummyless** line specifically.
   - Any work connecting **MBQC measurement-plane symmetries** (XY-plane reflection
     invariance) to error undetectability or to code symmetries.

## Search terms to try

`magic state injection MBQC graph state`, `gate teleportation Clifford skeleton`,
`HTTHTTHTTH Broadbent compilation`, `dummyless verification quantum`, `trappified scheme
stabiliser test`, `flag qubits error detection`, `cluster state error correction`,
`measurement-based quantum error correction`, `harmless / undetectable logical error
stabiliser`, `XY-plane measurement invariance MBQC`, `verification of quantum computation
Pauli twirl`.

## Known starting references (verify and build outward from these)

- A. Broadbent, *How to Verify a Quantum Computation*, Theory of Computing 14 (2018).
- J. Fitzsimons, E. Kashefi, *Unconditionally verifiable blind quantum computation*,
  PRA 96, 012303 (2017).
- D. Leichtle, L. Music, E. Kashefi, H. Ollivier, *Verifying BQP Computations on Noisy
  Devices with Minimal Overhead*, PRX Quantum 2, 040302 (2021).
- T. Kapourniotis, E. Kashefi, D. Leichtle, L. Music, H. Ollivier, *Unifying Quantum
  Verification and Error-Detection* (arXiv:2206.00631) — and the **dummyless SDQC** paper
  in that line (arXiv:2303.08865), which proves the `|V|-1` dummyless-generator result and
  the reflection-invariance ("harmless deviation") lemma.
- A. Broadbent, J. Nevin, *Noise-Robustness for Delegated Quantum Computation in the
  Circuit Model* (arXiv:2511.22844).

## Output format

Please return:
1. A short verdict on **Q1 (novelty)** and **Q2 (QEC/error-detection use)** — is the trick
   new, reused, or folklore; is it used in QEC/detection or not.
2. An annotated reference list (grouped by the areas in Q3), 1-2 sentences each on why it's
   relevant and what it actually shows.
3. A clearly separated **"gaps / could not find"** section listing claims you could not
   substantiate, so I know where the genuinely open / novel parts may be.

Do not pad with generic background; assume I know stabiliser formalism, MBQC, magic states,
and the QOTP. If a claim is uncertain, say so and give the closest real reference.
