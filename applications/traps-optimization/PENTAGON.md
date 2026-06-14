# Odd-cycle experiment: when the LP optimisation actually helps

**Takeaway:** the trap-distribution LP (Problem 1) is *particularly* useful when
the MBQC resource graph is **not bipartite**. On a bipartite graph a naive proper
colouring already reaches the optimal detection rate, so the LP gains nothing. On
a graph with an **odd cycle** the LP's *fractional* colouring strictly beats any
proper (integer) colouring, recovering the true optimum `1/χ_f`.

## Files
- `pentagon_circuit.py` — builds a 5-qubit circuit (CZ in a pentagon + single-qubit
  gates), transpiles it to MBQC, and verifies the resource graph; also exposes a
  veriphix `Client` via `make_client()`.
- `pentagon_experiment.py` — solves the LP on that graph and compares to a proper
  colouring. Outputs `results/pentagon_rates.json` and `plots/pentagon_rates.pdf`.

```bash
python applications/traps-optimization/pentagon_circuit.py     # build + verify graph
python applications/traps-optimization/pentagon_experiment.py  # run the comparison
```

## The circuit and its graph
- 5 qubits; `cz(i, (i+1) mod 5)` applied **first** → the five input nodes form a
  clean 5-cycle (pentagon);
- `rz`/`rx` single-qubit gates on each qubit → a chain ("tail") hanging off each
  pentagon node.

Verified resource graph: **pentagon (girth 5, non-bipartite)** with one tail per
node; each pentagon node has degree 3 (two cycle neighbours + one tail).

## Result (standard-trap pool, `ℰ = {Z_v}` on the pentagon)

| protocol | detection rate | what it is |
|---|---|---|
| **OptimizedTraps (LP)** | **0.400 = 2/5** | fractional colouring = `1/χ_f(C5)` |
| FK12 greedy | 0.333 = 1/3 | proper integer 3-colouring = `1/χ(C5)` |

The LP reaches the fractional optimum `2/5`; the proper colouring is stuck at
`1/3`. The optimal distribution is the uniform mixture over the five 2-node
independent sets `{0,2},{1,3},{2,4},{3,0},{4,1}` — a fractional 2.5-colouring.

## Why bipartite hides this and odd cycles expose it
The standard-trap detection rate equals `1/χ_f(G[noisy])`, the inverse fractional
chromatic number. Two regimes:

- **Bipartite graph** (e.g. the brickwork patterns): `χ_f = χ = 2`. Integer and
  fractional colouring coincide, so a proper 2-colouring already gives `1/2` and
  the LP shows **no improvement** — it only matches greedy. (This is exactly what
  the `optimize.py` brickwork run showed: LP `0.50` = bipartite FK12 `0.50`.)
- **Odd cycle present** (this pentagon): `χ_f < χ` (`5/2 < 3`). A proper colouring
  wastes a colour, but the LP's fractional distribution does not — so the LP
  **strictly beats** the proper colouring (`2/5 > 1/3`).

So the LP earns its keep precisely on non-bipartite resource graphs, where the
gap `χ ≠ χ_f` is real. On bipartite graphs the optimisation is redundant.

## Caveats
- Both rates are `≤ 1/2`: standard traps can never exceed `1/χ_f ≤ 1/2` (any edge
  caps a node pair at `1/2`). The odd cycle reveals the fractional improvement
  *below* `1/2`, it does not break the `1/2` ceiling.
- `Z` is a harmless deviation, so this is a **benchmarking / `1/χ_f`-matching**
  demonstration, not a security-against-harmful-deviations claim.
- `RandomTraps` reports `0.5` on this graph, but via a *different* (generalised
  multi-qubit) trap family — not the standard independent-set pool — so it is not
  the apples-to-apples comparison for the colouring story.
