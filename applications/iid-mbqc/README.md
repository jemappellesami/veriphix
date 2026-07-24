# i.i.d. MBQC benchmarking

Benchmarking a noisy MBQC device by treating trap rounds as a **statistical estimator**
rather than as an adversarial test. The device is modelled not as malicious but as an
unknown noise process that is identical and independent across rounds.

```
(Y, s) --Clopper-Pearson--> q_U --r<=kq--> p_U --majority vote--> eps_vote
                                                                    |
                                          eps_total = eps_bench + eps_vote
```

| symbol | meaning |
|---|---|
| `Y`, `s` | failed test rounds, test rounds run |
| `q` | probability a randomly chosen test round fails — **what the experiment measures** |
| `q_U` | upper confidence bound on `q`, valid except with probability `eps_bench` |
| `r` | probability a round carries a harmful error; `r <= k q` |
| `k` | number of test-run types — **2**, from the FK12 bipartite trap colouring |
| `c` | intrinsic error probability of the ideal computation (0 for a deterministic one) |
| `p_U` | per-round computation error bound, `c + (1-c) min(1, k q_U)` |
| `d` | computation rounds, majority-voted |

**The wall.** Majority voting only helps while `p_U < 1/2`, i.e. `q_U < alpha/k` with
`alpha = (1-2c)/(2-2c)`. At `k=2, c=0` that is `q_U < 0.25` — the same 0.25 the adversarial
analysis produces, since `alpha/k = alpha * detection_rate`. Note the bound is on `q_U`, not
on the measured `q_hat`: a tile whose raw rate clears the wall can still be uncertifiable if
`s` was too small. That gap is the whole point of figure D.

## Why this is worth doing

**No classical simulability anywhere.** A trap outcome is a stabilizer eigenvalue, known by
construction and computable in poly time at any size. XEB needs the ideal output
distribution and is therefore capped at what you can simulate — precisely failing in the
regime where benchmarking starts to matter. The instrument here does not degrade as the
device outgrows your ability to check it.

**A statement about a class, not an ensemble average.** Because the protocol is blind, the
server cannot tell a test round from a computation round, nor which computation is running.
So `q` measured on traps applies to **every MBQC computation on the same graph**. XEB
averages over a random-circuit ensemble; this quantifies over the whole class.

Worth being precise about which quantifiers are worst-case: over the computation (by
blindness), over which of the two test runs catches a harmful error (`r <= kq`), and over the
estimator's fluctuation (`q_U` rather than `q_hat`). All of it conditional on the noise being
i.i.d. and stationary — the one thing assumed rather than derived.

**Precision is nearly free; noise is not.** The round count goes as

```
d ~ log(1/eps) / (2 (1/2 - p_U)^2)
```

so the target precision enters only through a logarithm. Measured on the sweep in this
folder: at fixed noise, going from `eps = 1e-2` to `1e-9` — seven orders of magnitude — costs
11 rounds → 67. What is expensive is noise, where the cost diverges as `p_U -> 1/2`, and past
the wall no budget helps at all.

**The costing formula is dimension-free.** `d` depends on `q_U` and `eps` only; width and
depth appear nowhere in the analysis. They enter solely through the measured rate,
`(width, depth) -> q_U`. So there is one universal formula and one per-tile measurement, and
a 2x2 tile and a 6x10 tile are priced by the same expression with a different input. The
heatmap carries all the device-specific information; the pipeline carries none. That is what
lets the three applications below compose on top of a single sweep.

**It costs ~50x less than the adversarial analysis.** At matched effective error and
`eps = 1e-6`, this pipeline needs 33–173 rounds where `veriphix/util_rounds.py` needs
1368–12482. That gap is the price of dropping the i.i.d. assumption — now a measured number
rather than a rhetorical one, which is also the honest argument for stating the assumption
openly instead of hiding it.

## Files

| file | what it does | depends on |
|---|---|---|
| `iid_pipeline.py` | the statistics: Clopper–Pearson, `r<=kq`, exact majority tail, both driving modes | scipy only |
| `experiment.py` | MBQC simulation → per-tile `(n_fail, n_rounds)` CSV (single-process) | graphix, stim, veriphix |
| `experiment_dask.py` | the same sweep fanned out over Dask + SLURM; one CSV per `(p_ent, s)` | + dask, dask_jobqueue |
| `plot_heatmaps.py` | width×depth maps of `q_U`, required `d`, achieved `eps` | a results CSV |
| `verifiable_quantum_volume.py` | certification frontiers and volume metrics | a results CSV |
| `plot_pipeline.py` | parameter plots of the pipeline itself | **nothing** — pure theory |

One measurement, three applications:

| | fix | ask |
|---|---|---|
| **performance-driven** | `eps_target` | how many rounds does *this tile* cost? |
| **cost-driven** | `d` | what precision does *this tile* reach? |
| **verifiable quantum volume** | `d` and `eps` | *how much of the grid* is reachable at all? |

The first two are per-tile readings (`plot_heatmaps.py`); the third is a statement about the
device as a whole. `plot_pipeline.py` is deliberately independent of any experiment: it
characterises the analysis, not a device, and can be regenerated without a simulation.

## Re-running

Each stage is independent and re-runnable on its own.

**One tile by hand** — sanity-check the pipeline on a count:

```bash
./.venv/bin/python applications/iid-mbqc/iid_pipeline.py --failures 70 --rounds 1000
```

Add `--d 201` for the cost-driven reading. With no `--eps-bench` the `eps_bench`/`eps_vote`
split is optimised rather than guessed.

**The experiment** — laptop concept check (25 tiles, ~35 s):

```bash
./.venv/bin/python applications/iid-mbqc/experiment.py --widths 2,3,4,5,6 --depths 2,4,6,8,10 --rounds 2000
```

Or `--smoke` for a 4-tile, 2-second version. Results **append** and existing
`(p_ent, width, depth)` rows are skipped, so an interrupted sweep resumes and a grid can be
widened without recomputing. Cost per tile is two `compile_sampler` calls plus `rounds`
Clifford shots, so it grows with graph size (~`width*depth` nodes) and linearly in `--rounds`.

`experiment.py` is single-process: it keys resume on `(p_ent, width, depth)` and **ignores
`n_rounds`**, so appending a run at a different `--rounds` silently mixes two values of `s` in
one file (the script now warns before doing so — see the mixed-`s` note below). For anything
larger than a laptop check, use the cluster script instead.

**The experiment, on a cluster** — `experiment_dask.py` fans the tiles out over Dask + SLURM,
calling the same `simulate_tile`. It writes **one CSV per `(p_ent, rounds)`**, named
`mbqc_iid_p{p:.1e}_r{rounds}.csv`, so different shot counts can never share a file and the
mixed-`s` hazard is structurally impossible. Resume is per file.

```bash
# smoke-test the parallel path locally (LocalCluster, no SLURM):
./.venv/bin/python applications/iid-mbqc/experiment_dask.py --widths 2,3,4 --depths 2,4 --p-ents 1e-3,1e-2 --rounds 2000
```

```bash
# the real sweep on SLURM — 40x40 grid, 100k shots, three noise levels:
./.venv/bin/python applications/iid-mbqc/experiment_dask.py \
    --widths 1,2,3,...,40 --depths 1,2,3,...,40 --p-ents 1e-4,1e-3,1e-2 \
    --rounds 100000 --walltime 6 --memory 8 --cores 4 --port 8787 --scale 40
```

Sizing (extrapolated from the recorded timings): the slowest tile (40×40, ~6.5k nodes at
100k shots) is ~17 min; one noise level is ~117 CPU-h, so on `--scale 40` workers a full
40×40 grid is **~3 h wall per noise level (~9 h for three)**. `--walltime` must comfortably
exceed the slowest tile; 8 GB/job is ample. Noise levels are queued together — with the
cluster parallel there is nothing to gain from draining one before the next.

**Reproducibility caveat.** `base_seed` fixes each tile's circuit and traps, but the noise
realisation and Stim's shot sampling are drawn from unseeded RNGs, so `n_fail` is a fresh
sample every run, not bit-reproducible. This is statistically harmless — with `s` shots the
count is exactly what the Clopper–Pearson interval is built around — but a resumed job's
finished tiles keep their original draws while re-run tiles get new ones. To make it
bit-reproducible, seed the noise model's `rng` and pass a per-tile `seed` to Stim's sampler
inside `simulate_tile`.

**The plots:**

```bash
./.venv/bin/python applications/iid-mbqc/plot_heatmaps.py --csv applications/iid-mbqc/results/mbqc_iid.csv
```

```bash
./.venv/bin/python applications/iid-mbqc/plot_pipeline.py
```

**Verifiable quantum volume** — reads a results CSV, never re-runs the experiment:

```bash
./.venv/bin/python applications/iid-mbqc/verifiable_quantum_volume.py --d 101 --eps-target 1e-6
```

### How it works

The whole application is one number, `q_crit`, computed two ways.

**Forward** (`--d`, `--eps-target`). You have `d` rounds and want `eps_total <= eps_target`.
`eps_bench` is already spent on the confidence bound, leaving `eps_vote = eps_target -
eps_bench` for the vote. The majority error `majority_error(d, p_U)` increases with `p_U`,
which increases with `q_U`, so bisecting on `q_U` finds the largest noise bound that still
fits in `eps_vote`. That is `q_crit`: **the noisiest a tile may be and still be certifiable
at this budget.** A tile passes iff its own `q_U <= q_crit`. The boundary of the passing set
is the frontier; its extent is the volume.

**Inverse** (the default worked example). Rather than choosing `d` and `eps`, choose how much
of the grid you want to cover: `q_crit` is set to the 15th or 30th percentile of the tiles'
`q_U`, so by construction roughly that share falls below it. Then `rounds_for_noise` inverts
the same relation — `required_rounds(p_U(q_crit), eps - eps_bench)` — to report the minimum
`d` at each `eps` on the menu. This is the planning question: *"I want to cover a third of my
device's operating range; what does that cost me in rounds?"*

Both directions produce a `q_crit` and therefore a staircase on the same heatmap, so a
coverage target and a round budget can be read against each other directly. The percentile
is over discrete tiles, so a 15% request lands on whatever tile boundary is nearest — with 25
tiles it certified 4, i.e. 16%. The console reports the achieved fraction, not the requested
one.

### Reading `s` in the figure titles

Titles carry either `s=2000` or `mixed s`. `s` is the number of test rounds behind each
tile, and the label is `mixed s` when the tiles in one noise group were **not** all measured
with the same `s` — which the append-and-resume design makes easy to produce by accident
(run half a grid at `--rounds 2000`, widen it later at `--rounds 20000`, and the CSV now
holds both).

It is a warning, not a cosmetic note: `q_U` depends on `s`. A tile measured with fewer rounds
gets a wider Clopper–Pearson interval and hence a **higher** `q_U`, so it can fail
certification purely for being under-measured rather than for being noisy. A frontier drawn
across mixed-`s` tiles is therefore not a clean statement about the device, and the tiles are
not comparable to each other. If you see `mixed s`, re-run the sweep at a uniform `--rounds`
(or split the CSV by `n_rounds`) before reading anything into the frontier.

`plot_heatmaps.py` also reads the older sweeps in `applications/benchmark-stim*/` — it
accepts `n`/`t` for width/depth and `p_depol` for the noise column, and falls back to
reconstructing `Y = round(q_hat * s)` with `s` from the `_r<N>`/`_s<N>` filename when the
integer columns are missing. Prefer files carrying the integers: Clopper–Pearson is a
function of the count, and rounding a stored rate throws away the exactness it is chosen for.

## Figures

- **A** performance-driven: `d` vs measured `q_hat`, one curve per `eps_target`.
- **B** cost-driven: achieved `eps_total` vs `q_hat`, one curve per budget `d`. The `eps_bench`
  floor is visible as the flat left-hand region — past it, more computation rounds buy
  nothing and the budget should move to the benchmarking stage.
- **C** Clopper–Pearson vs Hoeffding: the bound, and what the difference costs in `d`.
  Hoeffding's slack is additive and independent of `Y`, so it charges the worst-case
  variance (`q=1/2`) even for a tile that failed twice in a thousand rounds — exactly the
  clean tiles you most want to certify.
- **D** the value of test rounds: `d` vs `s`, against the `s -> infinity` floor where `q` is
  known exactly. At `q_hat = 0.15`, `eps = 1e-6`: `s=1e3` costs 977 computation rounds where
  the floor is 131; `s=1e4` brings it to 207. That is the concrete form of "more shots means
  a more precise rate".
- **E** certification frontiers: the `q_U` grid with one staircase per budget. Everything
  inside a frontier is certifiable at that budget; the frontier *is* the verifiable quantum
  volume.
- **F** volume vs round budget: certified share of the grid, and largest certified
  width×depth, as `d` grows. Both curves plateau — past the plateau the remaining tiles are
  beyond the hard wall and no budget reaches them, which is the honest ceiling of the device
  at that noise level.
- **heat_qU / heat_d / heat_eps**: the width×depth grid, then the same grid read through the
  pipeline. Tiles past the wall are hatched.

`q_crit`, and therefore every frontier, depends on `eps_bench`: a tighter confidence level
raises every tile's `q_U` and shrinks the certified region. `eps_bench` is also a hard floor
on `eps_total`, so a target at or below it is unreachable at any `d` — the tables label that
case separately from "past the wall", because the fixes differ (re-run the benchmarking
stage vs. nothing will help).

## Assumptions and caveats

**Interleaving.** Under i.i.d. the test rounds could all be run as one block, and the
binomial estimate would be unchanged. But stationarity across the test block and the
computation block then becomes an assumption you are asserting rather than one blindness
enforces — a device that drifts breaks it. Interleaving costs nothing statistically (the
count is the same either way) and upgrades that assumption to a consequence of blindness.
What the i.i.d. setting drops is the hypergeometric/sampling-without-replacement analysis,
not the interleaving itself.

**`r <= k q` is worst-case** over which of the two test runs catches a harmful error. The
CSV records `n_fail_run0`/`n_fail_run1` separately (free — the rounds are already sampled per
run) so the tightness is measurable. If the two runs' rates come out close on real data,
there is a factor approaching 2 in `p_U` to reclaim, worth roughly 4× in `d` near the wall.

**Blinding is off in the simulation** (`Secrets(r=False, a=False, theta=False)`), because
theta-blinding would make the measurements non-Clifford and lose stim. The honest failure
rate is blinding-invariant, so this does not affect the measured `q`. But blindness is what
the *argument* for validity across computations rests on, so it is an assumption about the
real protocol, not something this simulation demonstrates.

**What this is not.** No guarantee against a malicious or time-dependent server. Drift,
correlated noise, or different noise during the computation block invalidates the inference.
The adversarial pipeline that does cover those cases is `veriphix/util_rounds.py`, and it
costs roughly 40–70× more rounds at the same effective error.
