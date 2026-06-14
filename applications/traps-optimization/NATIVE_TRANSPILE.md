# Native (non-brickwork) transpilation of the sampled circuits

**Premise tested:** transpiling the `.qasm` circuits with graphix's *native*
transpiler (instead of the brickwork transpiler) would give non-bipartite
resource graphs, making the LP advantage clearer.

**Result: the premise does not hold.** Native transpilation of these circuits
gives **bipartite** graphs (often trees) at every size tested (`3-5` … `7-8`,
0/40 non-bipartite each). So there is no fractional-chromatic gap to exploit.

## Why
The sampled circuits are built from **CNOT + single-qubit rotations** (`cx`,
`rx`), with **no direct CZ**. In graphix's native transpiler:

- a **CZ** maps to a single graph edge → a CZ-cycle *is* a graph cycle (odd
  cycles possible — this is what `pentagon_circuit.py` uses);
- a **CNOT** maps to a gadget with **ancilla nodes**, which doubles path lengths
  → an odd CNOT-ring becomes an **even** graph cycle.

Verified directly:

| structure | graph girth | bipartite? |
|---|---|---|
| CZ-ring(3) / CZ-ring(5) | 3 / 5 (odd) | **No** |
| CNOT-ring(3) / CNOT-ring(5) | 6 / 10 (even) | Yes |

So CNOT-based circuits can only produce even cycles → bipartite resource graphs.
Odd cycles require CZ gates, which the sampled circuits don't contain.

## What the experiment shows (`native_transpile_experiment.py`)
Per circuit: bipartite?, girth, and the standard-trap detection rate from the LP
vs FK12 greedy (`results/native_transpile.json`). Representative `3-5` output:

| circuit | nodes | bipartite | girth | LP | greedy |
|---|---|---|---|---|---|
| circuit001 | 11 | True | tree | 0.500 | 0.500 |
| circuit070 | 11 | True | tree | **0.500** | **0.333** |
| circuit072 | 11 | True | 6 | **0.500** | **0.333** |
| circuit106 | 17 | True | tree | **0.500** | **0.333** |
| … | | | | | |

`non-bipartite: 0/12`, but **LP beats greedy in 5/12**.

## The subtlety: two different "LP wins"
The LP-vs-greedy gaps above are **not** the fractional-chromatic effect (these
graphs are bipartite, `χ_f = χ = 2`). They happen because **greedy colouring is
itself suboptimal** — `nx.greedy_color` sometimes uses 3 colours where 2 suffice,
giving `1/3` instead of the true `1/2`. The LP always reaches `1/χ_f = 1/2`.

So there are two distinct advantages of the LP:

1. **Robustly reaching `1/χ_f`** even when a greedy/heuristic colouring stumbles
   — visible here (`0.5` vs `0.333`), but mundane: any optimal colouring also
   gets `1/2`.
2. **Beating *every* proper colouring** — only possible when `χ_f < χ`, i.e. on
   **non-bipartite** graphs (odd cycles). This is the fundamental win, and it
   needs CZ gates: see `PENTAGON.md` (`2/5 > 1/3`, where no proper colouring
   reaches `2/5`).

## Conclusion
Native transpilation of the CNOT+rotation `.qasm` circuits does **not** expose
the fractional advantage — the graphs are bipartite. The genuine "LP beats the
best possible colouring" demonstration requires odd cycles, which arise from
direct CZ gates (the pentagon circuit), not from CNOT-based circuits regardless
of the transpiler.

## Run
```bash
python applications/traps-optimization/native_transpile_experiment.py
python applications/traps-optimization/native_transpile_experiment.py --n-qubits 5 --depth 7
```
