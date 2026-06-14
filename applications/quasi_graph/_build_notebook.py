"""Generate quasi_graph_incompatibility.ipynb (no graphix/veriphix graph logic).

Run:  python _build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

md(r"""# Quasi-graph Clifford circuits: is the incompatibility graph bipartite?

This notebook experiments with **quasi-graph** Clifford circuits — the family described in
`applications/reviews/notes_verification.md`, *§ The case of a quasi-graph*.

A quasi-graph circuit is built from:

* **CZ** gates between qubits (the "graph" part), and
* **H-gadgets** — a Hadamard on a data qubit realised as a short CZ+H chain on 6 fresh ancillas.

We deliberately **drop all the graph-state machinery** of `veriphix` (no `graphix` patterns,
no flow, no `GraphStabilizer`). We only need a Clifford `C`, its tableau, and Pauli strings.
The circuit-building style is borrowed from
[`applications/detection_rate/bro-compil.py`](../detection_rate/bro-compil.py), with one change:
`bro-compil.py` realises the H-gadget with **CNOT**s, whereas the notes define it with **CZ** gates.
We use the **CZ** version from the notes so the circuit is genuinely "CZ + single-qubit" (quasi-graph).

### Goal
For a circuit `C`, the *canonical stabilizer basis* is `S_i = C† X_i C` for each wire `i`.
Two stabilizers are **compatible** (mergeable into one test run) iff they **commute on every
qubit index** — only then do they share a product-state +1-eigenstate. The **incompatibility
graph** has one vertex per wire and an edge between `i,j` when `S_i, S_j` are *not* compatible.

> **Conjecture (to test):** the incompatibility graph of a quasi-graph circuit is bipartite.

Spoiler from the experiment below: it is **not always** bipartite, but it is bipartite under a
precise, checkable condition.
""")

md("## 0. Imports")
code("""import itertools, random, time
import stim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
""")

md(r"""## 1. Circuit builder (CZ-based H-gadget, as in the notes)

H-gadget on data qubit `i` with 6 fresh ancillas `i1..i6` (notes §H-gadget):

$$\mathrm{CZ}_{i,i_1},\ H_{i_1},\ \mathrm{CZ}_{i_1,i_2},\ \mathrm{CZ}_{i_2,i_3},\ H_{i_3},\
  \mathrm{CZ}_{i_3,i_4},\ \mathrm{CZ}_{i_4,i_5},\ H_{i_5},\ \mathrm{CZ}_{i_5,i_6},\ H_{i_6}.$$

An *op* is either `("CZ", a, b)` or `("H_GADGET", i)`. We pad the circuit with an identity
across all wires so `stim` tracks every qubit (gadgets allocate ancillas lazily).""")
code('''Op = tuple  # ("CZ", a, b) | ("H_GADGET", i)

def build_circuit(n: int, ops: list[Op]) -> tuple[stim.Circuit, int]:
    """Build a quasi-graph circuit on `n` data qubits. Returns (circuit, total_qubits)."""
    lines: list[str] = []
    nxt = n  # next free ancilla index
    for op in ops:
        if op[0] == "CZ":
            lines.append(f"CZ {int(op[1])} {int(op[2])}")
        elif op[0] == "H_GADGET":
            i = int(op[1])
            a1, a2, a3, a4, a5, a6 = range(nxt, nxt + 6)
            lines += [
                f"CZ {i} {a1}", f"H {a1}",
                f"CZ {a1} {a2}", f"CZ {a2} {a3}", f"H {a3}",
                f"CZ {a3} {a4}", f"CZ {a4} {a5}", f"H {a5}",
                f"CZ {a5} {a6}", f"H {a6}",
            ]
            nxt += 6
        else:
            raise ValueError(f"unknown op {op!r}")
    # reference every wire so the tableau spans all `nxt` qubits
    lines = [f"I {' '.join(map(str, range(nxt)))}"] + lines
    return stim.Circuit("\\n".join(lines)), nxt
''')

md("""## 2. Tableau and canonical stabilizer basis

`stim` gives the forward tableau `T` of `C` (so `T.x_output(i) = C X_i C†`). We want the
**back-propagated** observable `S_i = C† X_i C`, i.e. the X-basis output measurement on wire `i`
pulled back to the input. That is exactly `(T⁻¹).x_output(i)`.""")
code('''def circuit_tableau(circuit: stim.Circuit) -> stim.Tableau:
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    return sim.current_inverse_tableau() ** -1   # forward tableau T of C

def canonical_basis(tab: stim.Tableau, N: int) -> list[stim.PauliString]:
    """S_i = C† X_i C for every wire i."""
    inv = tab ** -1
    return [inv.x_output(i) for i in range(N)]
''')

md(r"""## 3. The incompatibility graph (commute on **every** index)

Two single-qubit Paulis commute iff they are equal or at least one is the identity.
Two stabilizers are **compatible** iff this holds at *every* qubit index; otherwise there is
an index where one carries `X`/`Y`/`Z` that anticommutes with the other, and they are
**incompatible** (an edge).

(Note: globally, all `S_i` commute — they generate a stabilizer group. The *per-index* relation
is the non-trivial one that governs whether traps can be merged into a single product-state run.)""")
code('''def index_compatible(p: stim.PauliString, q: stim.PauliString) -> bool:
    """True iff p and q commute on every qubit index (mergeable into one product eigenstate)."""
    return all(not (a and b and a != b) for a, b in zip(list(p), list(q)))

def incompatibility_graph(stabs: list[stim.PauliString]) -> nx.Graph:
    N = len(stabs)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if not index_compatible(stabs[i], stabs[j]):
                G.add_edge(i, j)
    return G

def two_coloring(G: nx.Graph) -> dict:
    """Robust 2-coloring that also handles disconnected / isolated nodes."""
    color = {}
    for comp in nx.connected_components(G):
        color.update(nx.bipartite.color(G.subgraph(comp)))
    return color

def draw_incompat(G: nx.Graph, title: str) -> None:
    bip = nx.is_bipartite(G)
    if bip:
        c = two_coloring(G)
        node_color = ["#fc9272" if c.get(n, 0) else "#9ecae1" for n in G.nodes()]
    else:
        node_color = "#bdbdbd"
    pos = nx.spring_layout(G, seed=1)
    plt.figure(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color="#888", node_size=420)
    plt.title(f"{title}\\nbipartite = {bip}")
    plt.show()
''')

md(r"""## 4. Experiment — 3 data qubits + 1 H-gadget

"Interconnected with CZ" is ambiguous, and it turns out to matter. We first take a **line**
CZ structure `0–1–2` (a bipartite interaction graph) plus one H-gadget at the end.""")
code('''n = 3
ops = [("CZ", 0, 1), ("CZ", 1, 2), ("H_GADGET", 2)]
circuit, N = build_circuit(n, ops)
print(f"data qubits = {n}, total wires = {N}")

tab = circuit_tableau(circuit)
stabs = canonical_basis(tab, N)
print("\\nCanonical basis  S_i = C† X_i C:")
for i, s in enumerate(stabs):
    print(f"  S_{i} = {s}")

G = incompatibility_graph(stabs)
print(f"\\nincompatibility edges = {G.number_of_edges()}, bipartite = {nx.is_bipartite(G)}")
draw_incompat(G, "line CZ (0-1-2) + 1 H-gadget")
''')

md(r"""### The triangle breaks it

If instead we **fully interconnect** the 3 qubits with CZ (a triangle `K_3`), the incompatibility
graph contains an odd cycle and is **not** bipartite — even with no gadget.

Why: for a *pure* CZ circuit, `S_i = X_i \prod_{j\sim i} Z_j`, so `S_i, S_j` anticommute on some
index **iff `i~j`**. Hence the incompatibility graph *is exactly the CZ interaction graph*. A
triangle of CZs is an odd cycle, so it is not bipartite.""")
code('''ops_tri = [("CZ", 0, 1), ("CZ", 1, 2), ("CZ", 0, 2)]   # K3, no gadget
circ_tri, Ntri = build_circuit(3, ops_tri)
G_tri = incompatibility_graph(canonical_basis(circuit_tableau(circ_tri), Ntri))
print("triangle CZ: bipartite =", nx.is_bipartite(G_tri),
      "| odd cycle:", nx.cycle_basis(G_tri))
draw_incompat(G_tri, "triangle CZ (K3), no gadget")
''')

md(r"""## 5. Refined conjecture + scaling experiment

The line case is bipartite, the triangle is not. The H-gadget expands a single wire into a
short **path** of ancillas (itself bipartite), so the natural refined statement is:

> **Refined conjecture.** The incompatibility graph of a quasi-graph circuit is bipartite
> **iff the underlying CZ interaction graph on the data qubits is bipartite**; H-gadgets never
> break bipartiteness.

We test this over many random quasi-graph circuits, comparing `is_bipartite(incompat_graph)`
against `is_bipartite(data_CZ_graph)`.""")
code('''def random_quasi_graph(n, p_edge, n_gadgets, rng):
    edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < p_edge]
    cz = nx.Graph(); cz.add_nodes_from(range(n)); cz.add_edges_from(edges)
    ops = [("CZ", a, b) for a, b in edges]
    ops += [("H_GADGET", rng.randrange(n)) for _ in range(n_gadgets)]
    rng.shuffle(ops)
    return ops, cz

rng = random.Random(0)
trials = 600
mismatches = 0
for _ in range(trials):
    n = rng.randint(2, 5)
    ops, cz = random_quasi_graph(n, 0.5, rng.randint(0, 2), rng)
    circ, N = build_circuit(n, ops)
    G = incompatibility_graph(canonical_basis(circuit_tableau(circ), N))
    if nx.is_bipartite(G) != nx.is_bipartite(cz):
        mismatches += 1

print(f"trials = {trials}")
print(f"mismatches (incompat-bipartite  !=  data-CZ-bipartite) = {mismatches}")
print("=> refined conjecture holds" if mismatches == 0 else "=> refined conjecture FAILS")
''')

md("""## 6. Scale up a single (bipartite) instance — still runnable

Take a larger bipartite CZ structure (a grid is bipartite) plus several H-gadgets, and confirm
the incompatibility graph is bipartite and the whole pipeline runs fast.""")
code('''rows, cols = 3, 4                      # 12 data qubits, grid = bipartite
grid = nx.grid_2d_graph(rows, cols)
relabel = {node: i for i, node in enumerate(grid.nodes())}
grid = nx.relabel_nodes(grid, relabel)
n = grid.number_of_nodes()

rng = random.Random(7)
ops = [("CZ", a, b) for a, b in grid.edges()]
ops += [("H_GADGET", rng.randrange(n)) for _ in range(4)]   # 4 gadgets -> +24 wires
rng.shuffle(ops)

t0 = time.perf_counter()
circ, N = build_circuit(n, ops)
G = incompatibility_graph(canonical_basis(circuit_tableau(circ), N))
dt = time.perf_counter() - t0

print(f"data qubits = {n}, total wires = {N}, incompat edges = {G.number_of_edges()}")
print(f"data CZ grid bipartite      = {nx.is_bipartite(grid)}")
print(f"incompatibility bipartite   = {nx.is_bipartite(G)}")
print(f"elapsed = {dt*1000:.1f} ms")
draw_incompat(G, f"{rows}x{cols} grid + 4 H-gadgets  ({N} wires)")
''')

md(r"""## 7. Fixing the odd cycle: reduce the basis to the *relevant* wires

The triangle is a problem only because we insisted on a stabilizer for **every** wire. But a
**BQP** computation reads out **one** logical bit. We only need to certify that *one* output —
plus everything the logical qubit physically flows through.

**Role tracking.** Think of each initial qubit as a *role*; a wire is just whatever role it
currently carries. An H-gadget on the wire holding role `r` **teleports** `r` down its
6-ancilla chain: the role that enters on `w` exits on `a6`, so `a6` becomes the *new holder*
of `r` (a "swap" of which physical wire plays that role). The seven wires `{w, a1..a6}` all lie
on `r`'s trajectory.

Pick the first initial qubit as the **output role** (role `0`). The **relevant wires** are
exactly those that ever carried role `0`:

$$\mathcal R \;=\; \{\text{wire }0\}\ \cup\ \{\text{ancillas of every H-gadget applied to role }0\}.$$

The other initial qubits — and gadgets applied to *them* — never carry role `0`, so they are
irrelevant and we **drop their stabilizers** from the canonical basis. The builder below works
on *logical* ops (`("CZ", r, s)`, `("H", r)` on roles) and tracks the holder of each role.""")
code('''def build_quasi_graph(n: int, logical_ops: list) -> tuple[stim.Circuit, int, set, dict]:
    """Quasi-graph on `n` roles. logical_ops are ("CZ", r, s) | ("H", r) on ROLES.
    Returns (circuit, total_wires, relevant_wires_for_role0, role_trajectories)."""
    lines: list[str] = []
    holder = {r: r for r in range(n)}          # role -> current physical wire
    path = {r: [r] for r in range(n)}          # role -> wires it has occupied
    nxt = n
    for op in logical_ops:
        if op[0] == "CZ":
            r, s = op[1], op[2]
            lines.append(f"CZ {holder[r]} {holder[s]}")
        elif op[0] == "H":
            r = op[1]
            w = holder[r]
            a1, a2, a3, a4, a5, a6 = range(nxt, nxt + 6)
            nxt += 6
            lines += [
                f"CZ {w} {a1}", f"H {a1}",
                f"CZ {a1} {a2}", f"CZ {a2} {a3}", f"H {a3}",
                f"CZ {a3} {a4}", f"CZ {a4} {a5}", f"H {a5}",
                f"CZ {a5} {a6}", f"H {a6}",
            ]
            for a in (a1, a2, a3, a4, a5, a6):
                path[r].append(a)            # role r teleports through the chain ...
            holder[r] = a6                   # ... and now lives on a6
        else:
            raise ValueError(f"unknown op {op!r}")
    lines = [f"I {' '.join(map(str, range(nxt)))}"] + lines
    relevant = set(path[0])                  # every wire that carried the output role 0
    return stim.Circuit("\\n".join(lines)), nxt, relevant, path

def reduced_basis(tab: stim.Tableau, relevant: set) -> dict:
    """Canonical stabilizers S_i = C† X_i C, kept only for relevant wires i."""
    inv = tab ** -1
    return {i: inv.x_output(i) for i in sorted(relevant)}

def incompat_from_dict(stabs: dict) -> nx.Graph:
    keys = list(stabs)
    G = nx.Graph(); G.add_nodes_from(keys)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if not index_compatible(stabs[keys[a]], stabs[keys[b]]):
                G.add_edge(keys[a], keys[b])
    return G
''')

md("""### 7a. The triangle, reduced

Three roles in a CZ triangle. With **no** gadget on role 0, the relevant set is just `{0}`: one
stabilizer, no edges — **1-colorable**. The odd cycle simply isn't in the reduced problem
anymore, because we never look at roles 1 and 2.""")
code('''circ, N, R, _ = build_quasi_graph(3, [("CZ", 0, 1), ("CZ", 1, 2), ("CZ", 0, 2)])
tab = circuit_tableau(circ)

G_full = incompatibility_graph(canonical_basis(tab, N))          # all 3 wires
G_red = incompat_from_dict(reduced_basis(tab, R))                # relevant only

print(f"relevant wires R = {sorted(R)}")
print(f"full     : {G_full.number_of_nodes()} nodes, bipartite = {nx.is_bipartite(G_full)}")
print(f"reduced  : {G_red.number_of_nodes()} nodes, {G_red.number_of_edges()} edges, "
      f"bipartite = {nx.is_bipartite(G_red)}")
''')

md("""### 7b. Triangle + one H-gadget on the output role

Now apply an H-gadget to role 0. Its 6 ancillas join the relevant set; roles 1 and 2 stay out.
The reduced incompatibility graph is a short chain — bipartite, 2 colors.""")
code('''circ, N, R, path = build_quasi_graph(3, [("CZ", 0, 1), ("CZ", 1, 2), ("CZ", 0, 2), ("H", 0)])
tab = circuit_tableau(circ)
stabs = reduced_basis(tab, R)
G_red = incompat_from_dict(stabs)

print(f"role-0 trajectory = {path[0]}")
print(f"relevant wires R  = {sorted(R)}")
for i, s in stabs.items():
    print(f"  S_{i} = {s}")
print(f"\\nreduced: {G_red.number_of_nodes()} nodes, {G_red.number_of_edges()} edges, "
      f"bipartite = {nx.is_bipartite(G_red)}")
draw_incompat(G_red, "reduced graph: triangle + H-gadget on role 0")
''')

md("""### 7c. Does the reduction always kill the odd cycle?

Test over many random logical circuits (CZ between arbitrary roles, H-gadgets on **any** role):
is the *reduced* incompatibility graph always bipartite? We also track the greedy colour count.""")
code('''rng = random.Random(0)
trials = 600
nonbip = 0
max_colors = 0
for _ in range(trials):
    n = rng.randint(2, 5)
    lops = [("CZ", a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < 0.6]
    lops += [("H", rng.randrange(n)) for _ in range(rng.randint(0, 3))]
    rng.shuffle(lops)
    circ, N, R, _ = build_quasi_graph(n, lops)
    G = incompat_from_dict(reduced_basis(circuit_tableau(circ), R))
    if not nx.is_bipartite(G):
        nonbip += 1
    if G.number_of_nodes():
        max_colors = max(max_colors, len(set(nx.greedy_color(G).values())))

print(f"trials = {trials}")
print(f"reduced graph non-bipartite = {nonbip}")
print(f"max greedy colours used     = {max_colors}")
print("=> reduction removes the odd-cycle obstruction" if nonbip == 0 else "=> still non-bipartite somewhere")
''')

md(r"""## 8. Takeaways

* The **literal** conjecture ("the incompatibility graph is always bipartite") is **false** —
  a triangle of CZs is a counterexample.
* For a **pure** CZ circuit the incompatibility graph *equals* the CZ interaction graph
  (`S_i = X_i\prod_{j\sim i}Z_j`), so it is bipartite **iff that graph is bipartite**; an odd
  cycle (triangle) is the obstruction.
* H-gadgets expand a wire into a bipartite ancilla chain and never change the verdict on the
  *full* basis; the full graph is bipartite **iff the data-qubit CZ graph is bipartite**.
* **The fix (BQP single output).** We only need to certify one logical output. Tracking that
  role through the H-gadget teleportations and keeping stabilizers **only for the relevant
  wires** (`{output qubit} ∪ {its gadget ancillas}`) removes the irrelevant qubits — and with
  them the odd cycle. Over 600 random circuits the **reduced** incompatibility graph is always
  bipartite (≤ 2 colours); a bare triangle collapses to a single 1-colourable node.

Next: prove the reduced graph is always a (bipartite) chain along the output role's trajectory,
and relate this "relevant-wire" reduction to the `n-1` generators remark in the notes and to
`veriphix`'s `FK12` colouring strategy.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
with open("quasi_graph_incompatibility.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote quasi_graph_incompatibility.ipynb with", len(cells), "cells")
