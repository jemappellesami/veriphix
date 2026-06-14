# Problem 1 beyond fractional colouring: general deviations

## Why this experiment exists
The pentagon experiment (`PENTAGON.md`) is a **degenerate** instance of Problem 1
(arXiv:2206.00631): deviation set `{Z_v on every node}`, detection relation
`R[I, Z_v] = [v ∈ I]` (pure graph incidence). There the LP collapses to the
fractional-colouring LP and returns `1/χ_f` — a result already in the paper. It
shows nothing the LP adds *beyond* fractional colouring.

This experiment runs Problem 1 in its **general** form, where the LP has **no
graph-colouring interpretation** and is genuinely the right tool.

## The setup (`general_deviations.py`)
- **Resource graph:** the pentagon-with-tails (from `pentagon_circuit.py`).
- **General deviations `ℰ`:** a mix of
  - single-qubit *harmful* Paulis `X_v` and `Y_v` (both axes), and
  - multi-qubit *correlated* deviations `X_u X_v` on graph edges — after twirling,
    crosstalk-type hardware noise looks exactly like these 2-qubit Paulis.
- **General detection relation `R`:** the full anticommutation matrix between
  physical **X/Y-basis** traps (dummies allowed) and these deviations. This is an
  arbitrary 0/1 matrix — *not* vertex-in-independent-set incidence.

## What it shows
Sweeping the number of correlated 2-qubit deviations:

| #correlated | LP-optimised | non-adaptive (uniform) | gap |
|---|---|---|---|
| 0 | **1.0000** | 0.5000 | 0.500 |
| 1 | 0.6429 | 0.4545 | 0.188 |
| 2 | 0.6250 | 0.4545 | 0.171 |
| 3 | 0.5882 | 0.4545 | 0.134 |
| 4 | 0.5714 | 0.4545 | 0.117 |
| 5 | 0.5714 | 0.4545 | 0.117 |

(Plot: `plots/general_deviations.pdf`.)

Three points, none reducible to colouring:

1. **The optimum is not a fractional colouring.** For the full 15-deviation
   instance the LP's optimal distribution uses 6 tests with a **non-uniform mix of
   measurement bases** (X-basis mass `0.29`, Y-basis mass `0.71`) over
   *dummy-inclusive* test sets. No graph colouring produces this — the answer
   depends on the Pauli content of the deviations, not just the graph.

2. **The LP strictly beats a non-adaptive baseline** (uniform over the feasible
   traps) across the whole sweep. Crucially this gap is **not** a `χ`-vs-`χ_f`
   gap — both rates are below `1/χ_f` issues entirely. It is the value of
   *adapting the test distribution to the deviation structure*. The single-qubit
   case is the sharpest: the LP reaches `1.0` (it picks, per node, the basis that
   detects that node's harmful axis), while a fixed/uniform scheme is stuck at
   `0.5`.

3. **Graceful degradation.** Adding correlated errors lowers the achievable rate
   (2-qubit deviations are harder to catch), but the LP retains its advantage
   throughout.

## What this adds over the paper's result
The paper proves the LP exists and characterises the standard-trap case via
`1/χ_f`. This experiment is the **general-`R` regime in action**:

- it instantiates Problem 1 end-to-end on a real transpiled circuit, with a
  detection matrix built from actual `stim` anticommutation;
- it uses a deviation set that is **not** single-Pauli-per-node, so `R` is not
  graph incidence and the optimum is **not** a fractional colouring;
- it quantifies the LP's advantage as the gap over a non-adaptive distribution —
  the operational payoff of *optimising* the test distribution, separate from any
  chromatic-number argument.

This is the case where "solve the LP" is doing something a colouring search
cannot, and where a *learned / structured* deviation set (mixed Paulis,
correlations) genuinely shapes the optimal traps.

## Run
```bash
python applications/traps-optimization/general_deviations.py
open applications/traps-optimization/plots/general_deviations.pdf
```

## Caveats
- The baseline is "uniform over the feasible traps" (a fair non-adaptive
  reference); it is not identical to any named protocol.
- Pools use maximal independent sets in X/Y bases with neighbour-dummies, so the
  reported rates are lower bounds on the true generalised-trap optimum.
- `X`/`Y` are the harmful deviations (flip outcomes); `Z` is harmless and omitted.
