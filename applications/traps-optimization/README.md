# Trap-distribution optimisation experiments

Client-side optimisation of *which traps to run* against a learned noise model,
based on **Problem 1** of arXiv:2206.00631 ("Optimisation of the Distribution of
Tests"): given an error set ℰ, a feasible test pool ℋ, and the detection
relation `R`, find the distribution over tests maximising the worst-case
detection rate ε. Solved as a linear program (`scipy`), no quantum simulation.

Library pieces used (in `veriphix/`):
- `OptimizedTraps` protocol (`protocols.py`)
- `trap_optimization.py`: `build_detection_matrix`, `solve_trap_distribution`,
  `independent_set_pool`, `multi_basis_independent_set_pool`

All scripts are run from the repo root with the venv active.

## 1. Baseline optimisation on the learned heatmap — `optimize.py` / `plot.py`
Reads the observed trap-failure heatmap from `applications/noise_learning/results`,
builds ℰ = {Z_v : v noisy}, optimises the trap distribution, compares to FK12 /
RandomTraps, and draws the 3-panel "game" figure (learned noise / optimal trap
distribution / LP-dual adversary) + a detection-rate bar chart.

```
python applications/traps-optimization/run.py          # optimize + plot
open applications/traps-optimization/plots/game.pdf
open applications/traps-optimization/plots/detection_rates.pdf
```
Result on the (bipartite) brickwork: OptimizedTraps **0.50** vs FK12-greedy 0.33.
The standard-trap ceiling here is 1/χ_f = 1/2 (bipartite).

## 2. Biased vs two-axis harmful noise — `axis_sweep.py`
Sweeps graph structure × harmful-noise axis content (single-axis X, single-axis
Y, two-axis X+Y) over a physical X/Y-basis, neighbour-inclusive (dummy) pool.

```
python applications/traps-optimization/axis_sweep.py
open applications/traps-optimization/plots/axis_sweep.pdf
```
**Finding:** *single-axis* harmful noise (X-only or Y-only — biased / coherent
hardware) is detected with rate **1.0**. *Two-axis* harmful noise (X and Y, e.g.
depolarising) is **not** capped at 1/2: the **dummy mechanism** lifts it (0.6–1.0
here), because measuring a noisy node's neighbour places a `Z` on that node and
`Z` anticommutes both `X` and `Y`.

## 3. Connectivity sweep — `connectivity_sweep.py`
Grows the noisy region as a graph ball of radius r around a centre and plots the
*two-axis* detection rate vs. the region's internal edge count.

```
python applications/traps-optimization/connectivity_sweep.py
open applications/traps-optimization/plots/connectivity_sweep.pdf
```
With the dummy mechanism the rate stays well above 1/2 (≈0.66–1.0) across
connectivities.

## Theory (corrected)
Blindness confines trap **measurements** to the X-Y plane (the `+θ` padding is
undone by a pre-`Z(θ)` rotation, which only commutes through `CZ` in-plane).
But **dummies** (Z-eigenstate *preparations*) are allowed, and the graph
entanglement means measuring a node's neighbour places a `Z` on it. For a
generator-subset trap with measured set `c` and adjacency `Γ`:

| deviation at `v` | detected iff |
|---|---|
| `X_v` | `(Γc)_v = 1`  (odd measured neighbours) |
| `Y_v` | `c_v ⊕ (Γc)_v = 1` |
| `Z_v` (harmless) | `c_v = 1` |

A single trap catches **both** `X_v` and `Y_v` iff `c_v = 0, (Γc)_v = 1` — `v` is
a dummy with odd measured neighbours (on-site `Z`). So:

- **single-axis** harmful noise → `1.0` (fix the complementary basis / cover);
- **two-axis** harmful noise → reaches `1.0` iff a GF(2) **Z-cover** of the noisy
  region exists (`c_v=0, (Γc)_v=1` for all noisy `v`); otherwise lands strictly
  between `1/2` and `1`. The achievable value is a graph/region property — the
  same dummyless GF(2) machinery as `applications/detection_rate/`.

The naive `1/2` appears **only** if you refuse the dummy mechanism (measure noisy
nodes directly). There is **no** universal `1/2` ceiling for two-axis noise.

**Two earlier mistakes, corrected here:** (1) an unphysical `Z`-basis *measurement*
gave a bogus `2/3`; (2) a "`1/2` is a hard ceiling" claim ignored that a node
carries `Z` when its neighbour is measured (the dummy mechanism). Both are fixed
above; the pools used here still restrict to maximal independent sets, so the
reported rates are **lower bounds** on the true GF(2) optimum.

**Caveat:** this optimises against a *noise* threat (the device's actual errors),
not an adversarial server that could deviate outside ℰ — benchmarking / trusted-
but-noisy hardware, not unconditional security.
