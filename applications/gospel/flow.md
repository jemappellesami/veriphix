
# Gospel benchmark — pipeline documentation

---

## For humans

The goal of the Gospel benchmark is to use verification protocols as a **benchmark
for quantum computers**. The central question is: given a quantum computer with a
known noise model, can it actually run computations of a given size (width × depth)
in a way that is still verifiable? And if it can, at what cost in terms of rounds?

Concretely: the RandomTraps protocol requires the server to pass test rounds. Under
noise, test rounds fail more often. If too many rounds are needed to reach a given
security level ε, the machine is not practically useful for computations of that
size — even if it technically runs them. The heatmaps make this trade-off visible
across circuit dimensions and noise regimes.

### Step 1 — Circuit generation (`circuits_pipeline.py`)

For every (width, depth) pair in the sweep, generate a pool of random brickwork
circuits with `veriphix.sampling_circuits`. Then filter to **BQP-hard** circuits:
those whose acceptance probability `prob` satisfies `prob < bqp_error` or
`1 - prob < bqp_error`. These are the circuits where the BQP promise is tight —
the ones that actually stress the verification protocol.
The filtered subset is stored in `sampled_circuits/circuits-{n}-{d}-{bqp_tag}/`.

```
python applications/gospel/circuits_pipeline.py \
    --nqubits-min 3 --nqubits-max 6 \
    --depth-min 5   --depth-max 8   \
    --bqp-error 0.1 --n-shots 100
```

### Step 2 — Simulation (`simulation-cluster.py`)

For each sampled circuit and each noise level `p_ent` (entanglement depolarising
error, log-spaced from ~1e-5 to 1e-1), run `test_rounds=100` rounds of the
RandomTraps protocol under a `DepolarisingNoiseModel`. Record how many test rounds
fail. Results stream into `gospel_results_cluster.csv` as Dask futures complete,
so the job is resumable.

```
python applications/gospel/simulation-cluster.py          # local, all cores
python applications/gospel/simulation-cluster.py \        # SLURM cluster
    --walltime 2 --memory 8 --cores 4 --port 8787 --scale 20
```

CSV columns: `p_ent`, `width`, `depth`, `bqp_error`, `circuit_label`,
`traps_passed`, `nr_failed_test_rounds`, `test_rounds`.

### Step 3 — Plotting (`gospel_plots-v2-select.py`)

Aggregate the CSV by averaging `nr_failed_test_rounds / test_rounds` across
circuits for each `(p_ent, width, depth, bqp_error)` cell. Then plot two variants:

- **Discrete heatmap** — raw averaged failure rate on a depth × width grid. Cell
  values are annotated directly. Saved to `plots-select/`.
- **Continuous heatmap** — bicubic upsampling (10×) of the same grid for a smooth
  visual. Frontier lines are overlaid. Saved to `plots-select-continuous/`.

**Frontier lines** (continuous only): for a given round budget N and security
parameter ε, `util_rounds.maximize_robustness_under_budget` tells us the maximum
noise level `max_rho` the machine can have while still allowing the verifier to
reach security ε within N rounds. If a tile's failure rate is below `max_rho`, the
machine can handle that circuit size within budget — it is **feasible**. Above it,
the round cost becomes prohibitive. Multiple (N, ε) combinations are overlaid so
one can read off the trade-off: tighter security or larger N shifts the frontier.

Different noise regimes have different sensitivities: a low-noise machine sits in a
narrow failure-rate range where small changes in N or ε matter a lot, while a
high-noise machine needs very different N values to show any useful frontier.
For this reason, the sweep (N values, ε values) is configured **per p_ent** via
`FRONTIER_SWEEP` at the top of the file:

```python
FRONTIER_SWEEP: dict[float | None, tuple[list[int], list[float]]] = {
    6e-4: ([1800, 1850, 1900, 1950], [1e-7, 1e-8]),
    2e-3: ([4000, 5000, 6000],       [1e-7, 1e-8]),
    None: ([2000, 2500, 3000],       [1e-3, 1e-4]),  # default fallback
}
```

Keys are matched to actual data `p_ent` values in log-space. `None` is the fallback
for any `p_ent` not explicitly listed.

The `p_noise*` annotation in the top-right corner marks the most permissive
`max_rho` across the sweep used for that specific plot — the noise ceiling above
which no (N, ε) combination in that sweep is feasible.

```
python applications/gospel/gospel_plots-v2-select.py \
    --csv applications/gospel/gospel_results_cluster.csv
```

Edit `FILTER_BQP`, `FILTER_P_ENT`, and `FRONTIER_SWEEP` at the top of the file to
control which plots are generated and which frontiers are drawn.


Suggested caption for a LaTeX figure:
````
\caption{Average test-round failure rate as a function of circuit dimension
(width and depth), under depolarising noise on entangling gates
(see $p_{\mathrm{entangl}}$ in plot title).
Each frontier line, labelled $(N, \varepsilon)$, marks the boundary above which
the machine cannot achieve security $\varepsilon$ within $N$ total rounds:
circuit dimensions below a frontier are feasible for the corresponding budget,
those above are not.
$p_{\mathrm{noise}}^*$ is the highest tolerable failure rate across all
$(N, \varepsilon)$ combinations shown; for circuit dimensions whose failure rate
exceeds $p_{\mathrm{noise}}^*$, no combination in this sweep yields a feasible
solution.}
```

---

## For LLMs

**Repository context:** `veriphix` — a quantum verification framework.
This pipeline repurposes the RandomTraps verification protocol as a **benchmark for
quantum computers**: given a machine's noise model, can it run circuits of a given
size (width × depth) in a regime where verification is still practically feasible?
Feasibility is determined by the round budget required to reach a target security ε.

### Key concepts

- `p_ent`: entanglement depolarising error probability per gate (the machine's noise level).
- `bqp_error`: BQP hardness threshold. A circuit is BQP-hard if its acceptance
  probability is within `bqp_error` of 0 or 1. Only BQP-hard circuits are used.
- `p_failed_round = nr_failed_test_rounds / test_rounds`: the measured failure rate.
  Higher noise → more failed test rounds → more total rounds needed for security.
- `max_rho = w/s`: the maximum tolerable failure rate for a given `(c=bqp_error, N, ε)`.
  Computed by `veriphix.util_rounds.maximize_robustness_under_budget(c, detection_rate=0.5, epsilon_target, budget)`.
  A tile is **feasible** if `p_failed_round ≤ max_rho`: the machine can reach
  security ε for that circuit size within N total rounds.

### Files

| File | Role |
|---|---|
| `circuits_pipeline.py` | Generates and BQP-filters circuits into `sampled_circuits/` |
| `simulation-cluster.py` | Runs RandomTraps simulations, writes `gospel_results_cluster.csv` |
| `gospel_plots-v2-select.py` | Aggregates CSV, plots discrete + continuous heatmaps with frontiers |

### Data flow

```
circuits_pipeline.py
  → sampled_circuits/circuits-{n}-{d}-{bqp_tag}/   (QASM files + table.json)

simulation-cluster.py
  → gospel_results_cluster.csv
    columns: p_ent, width, depth, bqp_error, circuit_label,
             traps_passed, nr_failed_test_rounds, test_rounds

gospel_plots-v2-select.py
  → plots-select/heatmap_p{p_ent}_bqp{bqp}.pdf             (discrete)
  → plots-select-continuous/heatmap_p{p_ent}_bqp{bqp}.pdf  (continuous + frontiers)
```

### Plot internals

- Grid: `pivot(index=width, columns=depth, values=p_failed_round)`, width
  descending so larger circuits are at the top.
- Continuous: `scipy.ndimage.zoom(grid, 10, order=3)` after NaN-filling with
  `generic_filter`. Contours drawn with `ax.contour(x_cont, y_cont, smooth, levels=[max_rho])`.
- Frontier configs: one `(max_rho, color, label)` per `(N, ε)` pair.
  Color family = Blues/Greens/Purples/… per ε; shade = light→dark per N.
  Lines labelled inline with `clabel`; short-line fallback via `_place_label()`.
- `p_noise*` annotation: `max(max_rho for all frontier configs)`, placed at
  `ax.transAxes (0.98, 0.88)` top-right on the continuous plot.
- Output format: PDF (vector).
