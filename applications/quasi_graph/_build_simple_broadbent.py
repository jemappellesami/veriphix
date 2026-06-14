"""Generate broadbent_bipartite_demo.ipynb — a minimal, fast standalone demo.

Run:  python _build_simple_broadbent.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# Broadbent skeleton → bipartite trap graph (minimal demo)

Circuits are made of **CNOT** + **H-gadgets** (the Clifford skeleton of Broadbent's
`H = HTTHTTHTTH` compilation). Traps are `stab_q = G†Z_q G` (computational basis). Two traps are
**compatible** (mergeable into one test run) iff their stabilizers commute on every qubit index; an
edge marks an *incompatible* pair. Number of colours = number of test-run types.

**Claim:** the graph is always **bipartite (χ = 2)** — Broadbent's two test runs.""")

md("## Setup")
code('''import itertools, random
import stim, networkx as nx
import matplotlib.pyplot as plt

def tableau(circuit):
    sim = stim.TableauSimulator(); sim.do_circuit(circuit)
    return sim.current_inverse_tableau() ** -1                    # forward tableau G

def build_broadbent(n, lops):
    """CNOT data gates + CNOT-based H-gadget (role teleports to the 6th ancilla)."""
    lines, holder, nxt = [], {r: r for r in range(n)}, n
    for op in lops:
        if op[0] == "CNOT":
            lines.append(f"CNOT {holder[op[1]]} {holder[op[2]]}")
        else:
            d = holder[op[1]]; a1,a2,a3,a4,a5,a6 = range(nxt, nxt+6); nxt += 6
            lines += [f"H {d}", f"CNOT {a1} {d}", f"CNOT {a2} {a1}", f"H {a2}", f"CNOT {a3} {a2}",
                      f"CNOT {a4} {a3}", f"H {a4}", f"CNOT {a5} {a4}", f"CNOT {a6} {a5}", f"H {a6}"]
            holder[op[1]] = a6
    lines = [f"I {' '.join(map(str, range(nxt)))}"] + lines
    return stim.Circuit("\\n".join(lines)), nxt

def commute_each_index(p, q):
    return all(not (a and b and a != b) for a, b in zip(list(p), list(q)))

def trap_graph(circuit, N):
    invG = tableau(circuit) ** -1
    stab = {q: invG.z_output(q) for q in range(N)}               # G^dag Z_q G
    g = nx.Graph(); g.add_nodes_from(range(N))
    for i, j in itertools.combinations(range(N), 2):
        if not commute_each_index(stab[i], stab[j]): g.add_edge(i, j)
    return g
''')

md("## One circuit, drawn with its 2-colouring")
code('''lops = [("CNOT", 0, 1), ("H", 1), ("CNOT", 1, 2), ("H", 0), ("CNOT", 0, 2)]
circ, N = build_broadbent(3, lops)
g = trap_graph(circ, N)
print(f"wires = {N}, edges = {g.number_of_edges()}, bipartite = {nx.is_bipartite(g)}")

color = {}
for comp in nx.connected_components(g):
    color.update(nx.bipartite.color(g.subgraph(comp)))
plt.figure(figsize=(6, 4))
nx.draw(g, nx.spring_layout(g, seed=1), with_labels=True,
        node_color=["#fc9272" if color.get(q, 0) else "#9ecae1" for q in g.nodes()],
        edge_color="#888", node_size=400)
plt.title("Broadbent skeleton trap graph — two colours = two test runs")
plt.show()
''')

md("## Fast sweep — 300 random circuits")
code('''rng = random.Random(0)
non_bipartite = 0
for _ in range(300):
    n = rng.randint(2, 4)
    lops  = [("CNOT", a, b) for a, b in itertools.permutations(range(n), 2) if rng.random() < 0.4]
    lops += [("H", rng.randrange(n)) for _ in range(rng.randint(1, 2))]
    rng.shuffle(lops)
    circ, N = build_broadbent(n, lops)
    if not nx.is_bipartite(trap_graph(circ, N)):
        non_bipartite += 1

print(f"non-bipartite circuits: {non_bipartite} / 300")
print("=> always bipartite, chromatic number 2" if non_bipartite == 0 else "=> found a counterexample")
''')

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
with open("broadbent_bipartite_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote broadbent_bipartite_demo.ipynb with", len(cells), "cells")
