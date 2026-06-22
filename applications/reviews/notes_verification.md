# Veriphix paper
The paper must be preceeded by a review on verification protocols in prepare-and-send model, mostly based on the following works: [FK17, B18, LMKO21, KKLM22, KKLM23, BN25, SO26], and informal discussions. We aim first to synthethize that.

## Preliminaries
- MBQC, and MBQC in DQC
  - client prepares single-qubit states, and forwards angle
  - corrections are not needed, since they are absorbed in the angles
- Clifford and Pauli groups, Pauli noise on Pauli measurements
  - pauli noise just flips outcome or no, deterministically
- Quantum One Time Pad
  - consequence
  - propagation of the pad:
    - key update
    - measurement decoding
- Classical OTP, and hybrid pad
- BQP Computations, operationally: what do we do ?
  - BQP output predicate: guarantee that the probability distribution is biased
  - majority vote
- Blindness, operationally ?
  - the server must receive instructions that look indistinguishable
  - sensitive information about the computation must be kept on the Client side
  - client must then keep two things:
    - the initial computation, with the sensitive data
    - a "safe" version where the non-sensitive informations are leaked (order of operations, size of operations) and sensitive one are "abstracted" (i.e becomes "a gate of many possible gates")

## Blindness in DQC: how to hide information from a Server
> This section explains the `blindness` module and the blindness-related Client operations

1. Here we first explain the protocol.
    - using UBQC, the client has a MBQC pattern (with flow and XY pi/4 meas only), is ok with leaking G and order of measurement, but nothing else.
    - to delegate the pattern blindly, the client does the following:
      - blind the prepared states: use quantum one-time pad on the input states
      - blind the angles: add a classical kpi/4 padding on the rotation angle, and pre-rotate the qubits from the same amount. This compensate since Z rotations commute through CZ.
      - To hide this, actually all qubits must be blinded
      - re-state UBQC protocol
2. Implementation
    - Cleaning phase
      - the client thus creates a private DB containing the private info: input state preparation, measurement angles, info about the corrections
      - and the client prepares a public, clean pattern, containing the public info and abstract information about the sensitive part, i.e BaseN/BaseM instruction.
    - Secrets preparation phase: associate secrets to qubits and angles
    - Qubits preparation phase: prepare the states of the qubits to send, store a classical representation of the single-qubit states
    - then, when simulating the run:
      - receive outcomes from the server, decode them with knowledge of the secrets, like UBQC
      - compute the blinded angle as done in UBQC
      - continue
      - if quantum output, decode using the keys as in UBQC, otherwise everything is decoded measurement outcomes
3. Consequences
   - from the POV of the Server, blinded states and blinded instructions are received: nothing can be deduced
   - if client uses another privateDB and uses one with only Clifford angles, server receives the same thing. Take the all-zero pattern for reference, where there is no correction and all measurement angles are set to $0$, i.e all X-basis measurement. This pattern is indistinguishable from any other pattern on the same graph with same (partial) measurement order.
   - Any physical operation happening on the Server side is independent of the Client's secret parameters since they are blind. Hence, we can take the all-zero pattern above as reference, and consider that the Server's behavior on the all-zero pattern is representative of the Server's behavior on ALL the patterns sharing the same graph and measurement order.
   - The Client also has the freedom to choose different input and ancilla state. If the intent of the CLient is to do the target computation, the ancilla should be the typical $\ket +$ ones. Otherwise they might differ: if the Client wants to do a test then the ancilla states must be inferred by the type of test the Client wants to do.

## Pauli noise detection in Clifford circuits via stabilizer testing
### Main idea
We start with
- Clifford circuits on $n$ qubits. Let's call $\qubits$=`qubits` the set of qubits, actually $[n]$.
- We consider measurements in the $X$ basis.
- A Pauli noise applies before the measurements and we want to detect it. A good way to do so is via stabilizer testing:
  - fix the output measurement that you want to observe and check: it can be $X_i$ -> check outcome $b_i=0$, for $i\in\qubits$. Then we need to prepare a +1-eigenstate of $C^\dagger X_i C$ as input, aka interpret the conjugated Pauli string as a stabilizer, and prepare a stabilizer state. 
  - We can do other tests: $X_i X_j$ -> check $b_i\xor b_j=0$ for $i,j \in$ `qubits`. Then we need to prepare a +1-eigenstate of $X_iX_j$
  - We can do multiple checks. We actually need to do so until the set of stabilizers that are tested form a set of $n$ linearly independent stabilizers that generate the group $X_1, ..., X_n$.
  - Once we have such a set of tests, we realize we can compact some of them if the back-propagated stabilizers commute on each index, because only then they share a common +1-eigenstate that is tensor product of single-qubit states.
    - So we can make a single run made of different tests, and each test can be a subset of qubits of which we take the parity of the meausrement outcomes. 
    - We can thus merge and it becomes confusing so we need definitions

### Framework
Definitions:
- `qubits`: set of qubits on which the clifford acts. set(int)
- `trap`: set(int) contained in `qubits`
- `merge` method: if two traps are compatible
- `test_run`: fully parametrized by `set(trap)`. From it
  - a stabilizer (pauli string) is derived
  - we can generate a +1-eigenstate
  - we can analyze the outcomes: need to check the AND of "XOR i for i in trap" for trap in test_run.
- Let's define the canonical stabilizer basis as follows:
  - i -> C^\dagger X_i C
  - do this for all $i$
- undetected errors are Paulis that commute with all the stabilizers of the basis
- harmful errors are those containing at least one Z or Y.
- we can define a detection rate for a set of errors. it is epsilon if for all errors of that set, the probability (over test run sampled at random following the set of test runs) that the test run detects the deviation is > 1-epsilon.
```
\begin{equation}
  \forall \E\in \mathcal E:
  \Pr_{\run \sim \testruns}\left[
    \sum_{\trap\in \run}\left( |\trap \cap E|\mod 2 \right) >0
  \right] \geq 1-\epsilon
\end{equation}
```
### Strategies
#### Color-based traps
If we start with the canonical basis, it is trivial that we detect all the harmful errors. But doing one test per stab. in the canonical basis is a bit too much, so we'd like to merge. Unfortunately we can't merge everything, there is a condition that in general hard to solve.

We can draw an incompatibility graph: each stab on a vertex, then an edge between them if they are not compatible. An optimal merging procedure is thus an optimal coloring of the graph. These are color-based traps (one test run = one color) and it is NP-hard to find an optimal coloring.

So the procedure is as follows:
- find a coloring of the incompatibility graph
- for each color, create a test run made of all the nodes of that color. This is a set of compatible traps. -> it defines the test run as above (defines teh stabilizer, the traps check operation, we can generate +1 eigenstate)
#### Dummyless traps
Again start from the canonical basis. It detects anything harmful: yaay! but the stabilizers look like IXXIZIIYZIXXY, might be scary. From a practical perspective, we might be interested in stabilizers who only contain IXY, no Z. We call them dummyless. The reason is that any +1-eigenstate of such operator contains only states in the XY plane, as Z would create $\ket0$, that we don't want.

Good thing is: we can make changes of basis, by composing tests (xor of the outcome, product of the stab)! the question is thus: can we find another basis, made of $n$ linearly independent stabilizer, that generate the canonical basis, i.e they are made from products of stabs of the canonical basis, but now contain no Z ? This is the search for dummyless.

Note that it has been shown that we might not have to look for $n$ linearly indep stab, but $n-1$ might be enough, in MBQC for instance, where the Clifford circuit is a product of CZ only.
So the quest is: from the canonical basis, can you manipulate them to get a set of $n-1$ linearly independent stabilizers containing no $Z$ ? It is feasible in polynomial time (polyn. in $n$) in MBQC when the Clifford is a graph. Is it feasible for all Cliffords ? ^^
#### Random traps
here, forget about the basis forget about anything. RandomTraps consist in sampling an non-empty subset at random, and associating it a test run. In other terms, a test run here is just one trap, and this trap is a non-empty random subset. If we compute the detection rate, theoretically, it gives 1/2, same as if we had a bipartite incompatibility graph. quite amazing.

### The case of a Graph
- Color-based traps: the incompatibility graph is the graph itself, so the optimal merging procedure amounts to finding optimal coloring of the graph. Starting with computations on a bipartite graph is thus cool.
- Dummyless traps:
  - start with Rfull, the product of all stab of the basis. it contains no Z since all indices contain at least X -> gonna be X or Y
  - Add another stab to Rfull to build your next basis element. problem: you don't want a $Z$ to appear. so here is the realization: for even-degree nodes, taking Rfull * stab doesn't make a Z appear. So you do that for all even-degree nodes.
  - Then, for odd-degree nodes, you're not stuck. Take a neighboring node that is odd degree and take the product, it will work. If no odd degree neighboring, take an even degree node and find a neighboring odd degree node. In general, find odd-degree node pairs connected by even-degree nodes and take the stab alongside that chain. It will produce a dummyless stab.
  - You can build $n-1$ linearly indep. dummyless stabilizers that way!
  - runs in polynomial time
- Random traps: works straightforwardly.

### The case of a quasi-graph...
Quasi graph is:
- you make only CZ.
- Sometimes, you make a H gadget to do a Hadamard. It is like this
```
\paragraph{$\H$-gadget.} Let the gadget act on qubit $i$. Add $6$ ancilla qubits, that we call $i_1...i_6$. The $\H$-gadget is the following circuit: $\CZ_{i,i_1}, \H_{i_1}, \CZ_{i_1,i_2}, \CZ_{i_2,i_3}, \H_{i_3}, \CZ_{i_3,i_4}, \CZ_{i_4,i_5}, \H_{i_5}, \CZ_{i_5,i_6}, \H_{i_6}$.
```


- Color-based approach: yields bipartite graph
- random traps: works the same
- dummyless: is it possible ?

### Arbitrary cliffords:
- color-based traps: already mentioned, it is NP hard
- Randomtraps: works
- Dummyless: can we find a general polynomial time algorithm ?



## Reduction to Pauli deviations: when verification becomes stabilizer testing
To write later



## References

- **[FK17]** J. F. Fitzsimons and E. Kashefi, "Unconditionally verifiable blind quantum computation," *Phys. Rev. A* **96**, 012303 (2017). [doi:10.1103/PhysRevA.96.012303](https://doi.org/10.1103/PhysRevA.96.012303)

- **[B18]** A. Broadbent, "How to Verify a Quantum Computation," *Theory of Computing* **14**(11), 1–37 (2018). [doi:10.4086/toc.2018.v014a011](https://doi.org/10.4086/toc.2018.v014a011)

- **[LMKO21]** D. Leichtle, L. Music, E. Kashefi, and H. Ollivier, "Verifying BQP Computations on Noisy Devices with Minimal Overhead," *PRX Quantum* **2**, 040302 (2021).

- **[KKLM22]** T. Kapourniotis, E. Kashefi, D. Leichtle, L. Music, and H. Ollivier, "Unifying Quantum Verification and Error-Detection: Theory and Tools for Optimisations," arXiv:2206.00631 (2022).

- **[KKLM23]** T. Kapourniotis, E. Kashefi, D. Leichtle, L. Music, and H. Ollivier, "Asymmetric Quantum Secure Multi-Party Computation With Weak Clients Against Dishonest Majority," Cryptology ePrint Archive, Paper 2023/379 (2023). [eprint.iacr.org/2023/379](https://eprint.iacr.org/2023/379)

- **[BN25]** A. Broadbent and J. Nevin, "Noise-Robustness for Delegated Quantum Computation in the Circuit Model," arXiv:2511.22844 (2025).

- **[SO26]** S. Abdul Sater and H. Ollivier, "Composable Verification in the Circuit-Model via Magic-Blindness," arXiv:2601.07111 (2026).
