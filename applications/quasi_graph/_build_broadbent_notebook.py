"""Generate broadbent_skeleton_bipartite.ipynb.

Tells the full story: Broadbent's H-compilation -> our Clifford skeleton, the
investigation that went down a wrong path (X-basis + CZ + "all ancillas"), the
controlled experiment that isolated the confound, and the final result
(CNOT+H skeleton, Z-basis traps -> bipartite, chromatic number 2).

Run:  python _build_broadbent_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

md(r"""# Broadbent's Clifford skeleton and the two-test-run (bipartite) structure

This notebook collects the experiments connecting three things:

1. **Broadbent's verification protocol** (arXiv:1509.09180, *How to Verify a Quantum Computation*) and
   its `H = HPHPHPH = HTTHTTHTTH` compilation.
2. **Our H-gadget** Clifford skeleton (the CZ / CNOT gadget used in `quasi_graph_incompatibility.ipynb`).
3. **The magic-blindness paper** (arXiv:2601.07111, §4) which claims — *without proof* — that after
   Broadbent's compilation the trap-merging graph is **bipartite with chromatic number 2**, explaining
   Broadbent's two types of test runs.

**Headline result (§4):** for the Broadbent skeleton (CNOT + H), the trap incompatibility graph in the
*computational (Z) basis* used by the paper is **always bipartite, χ = 2** — verified over 1500 random
circuits.

**Important honesty note (§3):** an earlier version of these experiments reported *non-bipartite* graphs.
That was a **confounded experiment** — it changed the gate set (CZ), the gadget, *and* the measurement
basis (X) all at once. §3 isolates the cause and shows the non-bipartiteness belonged to the
**CZ + X-basis** family, *not* to Broadbent's CNOT + H skeleton. This notebook keeps that mistake visible
on purpose.
""")

md("## 0. Imports")
code("""import itertools, random
import stim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
""")

# ---------------------------------------------------------------- shared helpers
md(r"""## 0a. Shared helpers

`tableau(circuit)` returns the forward Clifford tableau `G`. The **canonical trap basis** is
`stab_q = G† P_q G`, where `P = Z` is the paper's computational-basis measurement (and `P = X` is the
MBQC convention we used in the other notebook — we compare both below). Two traps are **compatible**
(mergeable into one test run) iff their stabilizers **commute on every qubit index**; an incompatibility
graph edge marks a pair that is *not* compatible. Merging = graph colouring, so χ = number of test-run
types.""")
code('''def tableau(circuit: stim.Circuit) -> stim.Tableau:
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    return sim.current_inverse_tableau() ** -1            # forward tableau G

def index_compatible(p: stim.PauliString, q: stim.PauliString) -> bool:
    """Commute on every qubit index <=> share a product +1-eigenstate."""
    return all(not (a and b and a != b) for a, b in zip(list(p), list(q)))

def trap_graph(invG: stim.Tableau, wires, basis: str) -> nx.Graph:
    """Incompatibility graph of traps stab_q = G^dag P_q G for q in wires (P = X or Z)."""
    out = invG.z_output if basis == "Z" else invG.x_output     # invG = G^-1 = G^dag
    stab = {q: out(q) for q in wires}
    G = nx.Graph(); G.add_nodes_from(wires)
    ks = list(wires)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if not index_compatible(stab[ks[i]], stab[ks[j]]):
                G.add_edge(ks[i], ks[j])
    return G
''')

# ================================================================ Section 1
md(r"""## 1. Provenance: Broadbent's compilation → our Clifford skeleton

Broadbent realises a Hadamard with the identity (her eqn. for the H-gadget)
$$\mathrm H = \mathrm H\mathrm P\mathrm H\mathrm P\mathrm H\mathrm P\mathrm H, \qquad \mathrm P = \mathrm T^2,$$
so $\mathrm H = \mathrm H\mathrm T\mathrm T\mathrm H\mathrm T\mathrm T\mathrm H\mathrm T\mathrm T\mathrm H$
(**6 T's, 4 H's**). Each $\mathrm T$ is **magic-state injection**: an auxiliary qubit + a CNOT +
a measurement (the $\mathrm P^x$ correction vanishes in test runs). Dropping the rotations leaves the
**Clifford skeleton**: 6 ancillas, 6 CNOTs, 4 Hadamards — exactly our gadget.

We verify (a) the CZ-gadget and the CNOT-gadget agree on the output-side stabilizers, and
(b) the gadget realises the **identity teleportation** that a test run requires (since in a test run the
$\mathrm P$'s are identity, `HPHPHPH → H⁴ = I`).""")
code('''cz_gadget   = "CZ 0 1\\nH 1\\nCZ 1 2\\nCZ 2 3\\nH 3\\nCZ 3 4\\nCZ 4 5\\nH 5\\nCZ 5 6\\nH 6"
cnot_gadget = "H 0\\nCNOT 1 0\\nCNOT 2 1\\nH 2\\nCNOT 3 2\\nCNOT 4 3\\nH 4\\nCNOT 5 4\\nCNOT 6 5\\nH 6"

for name, s in [("CZ-gadget", cz_gadget), ("CNOT-gadget", cnot_gadget)]:
    invG = tableau(stim.Circuit(s)) ** -1
    print(f"{name:11s}:  C^dag X_6 C = {invG.x_output(6)}   C^dag Z_6 C = {invG.z_output(6)}")
print("\\n-> identical output-side stabilizers: the CNOT->CZ (H CNOT H = CZ) rewrite is faithful.")
''')
code('''# (b) identity-teleportation check for the CZ gadget (statevector)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
H = (X + Z) / np.sqrt(2); plus = np.array([1, 1], complex) / np.sqrt(2)
N = 7
def kron(ops):
    o = ops[0]
    for x in ops[1:]: o = np.kron(o, x)
    return o
def g1(g, q): return kron([g if i == q else I2 for i in range(N)])
def cz(a, b):
    d = np.ones(2 ** N, complex)
    for idx in range(2 ** N):
        bits = [(idx >> (N - 1 - k)) & 1 for k in range(N)]
        if bits[a] == 1 and bits[b] == 1: d[idx] = -1
    return np.diag(d)
U = np.eye(2 ** N, dtype=complex)
for op in [("CZ",0,1),("H",1),("CZ",1,2),("CZ",2,3),("H",3),("CZ",3,4),("CZ",4,5),("H",5),("CZ",5,6),("H",6)]:
    U = (cz(op[1], op[2]) if op[0] == "CZ" else g1(H, op[1])) @ U
def channel(psi):
    state = U @ kron([psi] + [plus] * 6)
    red = kron([plus.conj()] * 6 + [I2]) @ state         # wires 0..5 measured in X (outcome +)
    return red / np.linalg.norm(red)
for label, psi in [("|0>", np.array([1,0],complex)), ("|1>", np.array([0,1],complex)),
                   ("|+>", plus), ("rand", np.array([0.6,0.8j],complex))]:
    o = channel(psi)
    cands = {"I":psi,"X":X@psi,"Z":Z@psi,"H":H@psi}
    match = [k for k,v in cands.items() if abs(abs(np.vdot(v/np.linalg.norm(v), o)) - 1) < 1e-6]
    print(f"in {label:5s}: output (test-run channel d->a6) = {match}")
print("\\n-> random input matches only 'I': the skeleton is the identity teleportation a test run checks.")
''')

# ================================================================ Section 2
md(r"""## 2. The investigation, and the wrong turn

We started (other notebook) from **CZ** circuits with **X-basis** traps `C† X_i C`, plus a BQP
reduction to a set of "relevant" wires. Two reduction rules were tried:

* **worldline**: output qubit + ancillas of gadgets *on the output role* — always bipartite;
* **all ancillas**: output qubit + *every* ancilla — gave a **non-zero** non-bipartite count.

The exact count depends on the circuit generator (the original interleaved generator gave ~94/777; the
simpler generator below gives a smaller but still non-zero number). At the time this looked like a real
obstruction. The cell below reproduces it.""")
code('''def build_cz_roles(n, lops):
    """CZ data gates + CZ H-gadget, role teleportation. Returns circ, N, all_ancillas, role0_worldline."""
    lines=[]; holder={r:r for r in range(n)}; path={r:[r] for r in range(n)}; nxt=n; anc=set()
    for op in lops:
        if op[0]=="CZ":
            lines.append(f"CZ {holder[op[1]]} {holder[op[2]]}")
        else:
            r=op[1]; w=holder[r]; a=list(range(nxt,nxt+6)); nxt+=6; a1,a2,a3,a4,a5,a6=a
            lines+=[f"CZ {w} {a1}",f"H {a1}",f"CZ {a1} {a2}",f"CZ {a2} {a3}",f"H {a3}",
                    f"CZ {a3} {a4}",f"CZ {a4} {a5}",f"H {a5}",f"CZ {a5} {a6}",f"H {a6}"]
            anc.update(a)
            for x in a: path[r].append(x)
            holder[r]=a6
    lines=[f"I {' '.join(map(str,range(nxt)))}"]+lines
    return stim.Circuit("\\n".join(lines)), nxt, anc, set(path[0])

rng=random.Random(11); T=777; nb_world=nb_all=0
for _ in range(T):
    n=rng.randint(2,5)
    lops=[("CZ",a,b) for a,b in itertools.combinations(range(n),2) if rng.random()<0.5]
    lops+=[("H",rng.randrange(n)) for _ in range(rng.randint(1,3))]
    rng.shuffle(lops)
    c,N,anc,world=build_cz_roles(n,lops); invG=tableau(c)**-1
    if not nx.is_bipartite(trap_graph(invG, sorted({0}|world), "X")): nb_world+=1
    if not nx.is_bipartite(trap_graph(invG, sorted({0}|anc),   "X")): nb_all  +=1
print(f"CZ circuits, X-basis traps, {T} trials:")
print(f"   worldline rule  (output + role-0 ancillas):  non-bipartite = {nb_world}")
print(f"   all-ancillas rule (output + ALL ancillas)  :  non-bipartite = {nb_all}   <-- looked like an obstruction")
''')

md(r"""### What was actually wrong

Comparing this to the Broadbent result later, **three things differed at once**:

| | "obstruction" run (§2) | Broadbent skeleton (§4) |
|---|---|---|
| data gates | **CZ** | **CNOT** |
| H-gadget | CZ-based | CNOT-based |
| measurement basis | **X** (`C†X_iC`) | **Z** (`G†Z_qG`) |

Changing three variables together and attributing the effect to one of them is the methodological error.
§3 fixes it with a controlled experiment.""")

# ================================================================ Section 3
md(r"""## 3. Controlled experiment — isolate gate set vs. measurement basis

We generate one random *logical* structure and realise it **both** as a CNOT circuit and a CZ circuit,
then build the trap graph in **both** the X- and Z-bases. Four conditions, identical structure.""")
code('''def build_gateset(n, lops, gate):
    """Same logical structure realised with `gate` in {"CNOT","CZ"} (data gates + H-gadget)."""
    lines=[]; holder={r:r for r in range(n)}; nxt=n
    for op in lops:
        if op[0]=="ENT":
            lines.append(f"{gate} {holder[op[1]]} {holder[op[2]]}")
        else:
            r=op[1]; d=holder[r]; a=list(range(nxt,nxt+6)); nxt+=6; a1,a2,a3,a4,a5,a6=a
            if gate=="CNOT":
                lines+=[f"H {d}",f"CNOT {a1} {d}",f"CNOT {a2} {a1}",f"H {a2}",f"CNOT {a3} {a2}",
                        f"CNOT {a4} {a3}",f"H {a4}",f"CNOT {a5} {a4}",f"CNOT {a6} {a5}",f"H {a6}"]
            else:
                lines+=[f"CZ {d} {a1}",f"H {a1}",f"CZ {a1} {a2}",f"CZ {a2} {a3}",f"H {a3}",
                        f"CZ {a3} {a4}",f"CZ {a4} {a5}",f"H {a5}",f"CZ {a5} {a6}",f"H {a6}"]
            holder[r]=a6
    lines=[f"I {' '.join(map(str,range(nxt)))}"]+lines
    return stim.Circuit("\\n".join(lines)), nxt

rng=random.Random(42); T=1500
nonbip={(g,b):0 for g in ("CNOT","CZ") for b in ("X","Z")}
edges ={(g,b):0 for g in ("CNOT","CZ") for b in ("X","Z")}
for _ in range(T):
    n=rng.randint(2,5)
    lops=[("ENT",a,b) for a,b in itertools.permutations(range(n),2) if rng.random()<0.35]
    lops+=[("H",rng.randrange(n)) for _ in range(rng.randint(1,3))]
    rng.shuffle(lops)
    for gate in ("CNOT","CZ"):
        c,N=build_gateset(n,lops,gate); invG=tableau(c)**-1
        for basis in ("X","Z"):
            g=trap_graph(invG, range(N), basis)
            if not nx.is_bipartite(g): nonbip[(gate,basis)]+=1
            edges[(gate,basis)]+=g.number_of_edges()

print(f"Controlled 2x2 (identical logical structures), {T} trials, all wires:\\n")
print(f"{'gate / basis':18s}{'non-bipartite':>15s}{'avg edges':>12s}")
for gate in ("CNOT","CZ"):
    for basis in ("X","Z"):
        print(f"{gate+' / '+basis+'-meas':18s}{nonbip[(gate,basis)]:>15d}{edges[(gate,basis)]/T:>12.1f}")
''')

md(r"""### Reading the 2×2

* **CNOT (Broadbent's gate set) is bipartite in *both* bases** — with non-trivial graphs (avg ~15 edges
  in Z). The two-colourability is real, not a basis artefact.
* The non-bipartiteness belongs to **CZ + X-basis** only.
* **CZ + Z-basis** is bipartite but *trivially* (avg 0 edges: every `G†Z_qG` is a pure-Z string), so it
  carries no information.

**Mechanism.** The Hadamard is the *only* gate that exchanges X↔Z. CNOT preserves type per wire
(`Z_c→Z_c, Z_t→Z_cZ_t`; `X_c→X_cX_t, X_t→X_t`), so each trap's Pauli at every index is fixed by Hadamard
parity → two classes → bipartite, in either basis. **CZ itself mixes** (`X_a→X_aZ_b`); in a basis where
that mixing is visible (X) it can close odd cycles. Broadbent compiles to **CNOT + H**, the non-mixing
regime.""")

# ================================================================ Section 4
md(r"""## 4. Final experiment — Broadbent skeleton, Z-basis traps → χ = 2

This is the experiment that backs the paper's claim. Circuits = random CNOTs between data wires +
CNOT-based H-gadgets (the Broadbent skeleton). Traps = `stab_q = G† Z_q G` (the paper's computational
basis). We report bipartiteness and the chromatic number.""")
code('''def build_broadbent(n, lops):
    """Broadbent skeleton: CNOT data gates + CNOT-based H-gadget, role teleportation."""
    lines=[]; holder={r:r for r in range(n)}; nxt=n; anc=set()
    for op in lops:
        if op[0]=="CNOT":
            lines.append(f"CNOT {holder[op[1]]} {holder[op[2]]}")
        else:
            r=op[1]; d=holder[r]; a=list(range(nxt,nxt+6)); nxt+=6; a1,a2,a3,a4,a5,a6=a
            lines+=[f"H {d}",f"CNOT {a1} {d}",f"CNOT {a2} {a1}",f"H {a2}",f"CNOT {a3} {a2}",
                    f"CNOT {a4} {a3}",f"H {a4}",f"CNOT {a5} {a4}",f"CNOT {a6} {a5}",f"H {a6}"]
            anc.update(a); holder[r]=a6
    lines=[f"I {' '.join(map(str,range(nxt)))}"]+lines
    return stim.Circuit("\\n".join(lines)), nxt, anc

rng=random.Random(0); T=1500; non_bip=0; chrom_two=0; max_edges=0
for _ in range(T):
    n=rng.randint(2,5)
    lops=[("CNOT",a,b) for a,b in itertools.permutations(range(n),2) if rng.random()<0.35]
    lops+=[("H",rng.randrange(n)) for _ in range(rng.randint(1,3))]
    rng.shuffle(lops)
    c,N,anc=build_broadbent(n,lops); invG=tableau(c)**-1
    g=trap_graph(invG, range(N), "Z")
    if not nx.is_bipartite(g): non_bip+=1
    if g.number_of_edges()>0 and nx.is_bipartite(g): chrom_two+=1
    max_edges=max(max_edges,g.number_of_edges())
print(f"Broadbent skeleton (CNOT + H-gadgets), Z-basis traps, {T} random circuits:")
print(f"   non-bipartite                        = {non_bip}")
print(f"   bipartite with >=1 edge (so chi = 2) = {chrom_two}")
print(f"   max edges seen                       = {max_edges}")
''')

md("### A concrete instance, drawn with its 2-colouring")
code('''rng=random.Random(3)
n=4
lops=[("CNOT",a,b) for a,b in itertools.permutations(range(n),2) if rng.random()<0.4]
lops+=[("H",rng.randrange(n)) for _ in range(2)]
rng.shuffle(lops)
c,N,anc=build_broadbent(n,lops); invG=tableau(c)**-1
g=trap_graph(invG, range(N), "Z")
print(f"wires={N}, edges={g.number_of_edges()}, bipartite={nx.is_bipartite(g)}")

color={}
for comp in nx.connected_components(g): color.update(nx.bipartite.color(g.subgraph(comp)))
node_color=["#fc9272" if color.get(q,0) else "#9ecae1" for q in g.nodes()]
plt.figure(figsize=(6,5))
nx.draw(g, nx.spring_layout(g,seed=2), with_labels=True, node_color=node_color,
        edge_color="#888", node_size=420)
plt.title(f"Broadbent skeleton, Z-basis traps: two colours = Broadbent's X-test / Z-test")
plt.show()
''')

# ================================================================ Section 5
md(r"""## 5. Why it *must* be 2 — connection to the magic-blindness paper

arXiv:2601.07111 §4 states, without proof:
> *It can be shown that the graph corresponding to the traps merging here is a bipartite graph, with a
> chromatic number of 2, which explains the protocol with two types of test runs.*

Two proofs, consistent with §3 and §4:

**Top-down (the clean one).** Broadbent explicitly builds **two** global product input states — the
X-test and the Z-test. Each one simultaneously stabilises a whole *set* of useful-qubit traps. By the
paper's compatibility definition (shared product +1-eigenstate ⟺ stabilizers commute per index), each
test is a **clique of mutually compatible traps** = one colour class. Two test states ⟹ two colour
classes ⟹ χ = 2.

**Bottom-up (the mechanism).** H is the unique X↔Z exchanger; CNOT preserves type. So every trap's Pauli
at each index is set by Hadamard parity → exactly two trap types → bipartite. This is why the only way to
break it (§3) was to introduce a *second* mixer, CZ, and look in the X-basis.

So "it has to be 2 because it is the same skeleton" is correct: the two-colouring already exists, fully
built, as Broadbent's two test runs.""")

# ================================================================ Section 6
md(r"""## 6. Takeaways

* **Provenance (§1):** our H-gadget *is* Broadbent's `HTTHTTHTTH` Clifford skeleton — 6 ancillas, 6 CNOTs,
  4 Hadamards — and it realises the identity teleportation a test run checks. CZ↔CNOT differ only by the
  ancilla basis, which we disregard.
* **The wrong turn (§2):** an apparent "non-bipartite obstruction" came from a **confounded** setup
  (CZ + CZ-gadget + X-basis). It was never about Broadbent's skeleton.
* **The fix (§3):** a controlled 2×2 shows **CNOT circuits are bipartite in both bases**; the
  non-bipartiteness is the **CZ + X-basis** family. The Hadamard being the unique X↔Z mixer is the cause.
* **The result (§4):** for the **Broadbent skeleton (CNOT + H) with Z-basis traps**, the incompatibility
  graph is **always bipartite with χ = 2** (1500/1500), confirming the paper's unproven claim.
* **The proof (§5):** χ = 2 because Broadbent's two test states are the two colour classes.

**Caveats (open):** circuits here are CNOT + H-gadget *proxies*, not full Broadbent-compiled circuits
(no explicit T→MSI compilation); traps are taken on **all** wires, a superset of the paper's useful
qubits `Q = {output} ∪ {MSI ancillas}`. Bipartite on all wires ⟹ bipartite on `Q`, so this is stronger,
but a genuinely compiled circuit restricted to `Q` would make it airtight.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
with open("broadbent_skeleton_bipartite.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote broadbent_skeleton_bipartite.ipynb with", len(cells), "cells")
