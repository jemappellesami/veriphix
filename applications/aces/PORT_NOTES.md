# ACES port notes

Porting the legacy ACES (Averaged Circuit Eigenvalue Sampling) experiment to the
*current* checkout of this repo. Two surfaces changed, not one:

* **veriphix** — the verification API was restructured (`TrappifiedCanvas` → `TestRun`,
  `Secrets`/`remove_flow` moved, `create_test_runs`/`delegate_test_run` replaced by a
  `VerificationProtocol` object + `Client.execute_test_run`).
* **graphix** (installed `0.3.6.dev30+g6dd835a79`) — the noise-model API and the
  state/measurement representation changed. The provided `noise model` and
  `stim pauli preprocessing` files target an *older* graphix and do **not** import as-is.

`gospel` is **not** used. Per the repo owner:
* brickwork transpiler → `veriphix.sampling_circuits.brickwork_state_transpiler`
* uncorrelated depolarising noise model → added to `veriphix/` alongside the other models
* stim pauli preprocessing → vendored into this folder (`stim_pauli_preprocessing.py`)
* dask → dropped (no parallelisation for now)

How each mapping was confirmed: `R` = read the installed source, `T` = exercised at
runtime in this environment, `X` = repo's own working code (`malicious_noise_model.py`,
`gaussian_region_noise_model.py`, `client.py`) shows the current idiom.

## veriphix

| Legacy symbol | New name / path | Confirmed |
|---|---|---|
| `from veriphix.client import Secrets` | `from veriphix.blinding import Secrets` | R `blinding.py:32` |
| `Secrets(r=False, a=False, theta=False)` | unchanged constructor (blinding off) | R, T |
| `from veriphix.client import remove_flow` | unchanged path; **behaviour changed** (see below) | R `client.py:75`, T |
| `from veriphix.client import Client` | unchanged path | R `client.py:98`, T |
| `Client(pattern, secrets)` | `Client(pattern=pattern, secrets=..., protocol=FK12(manual_colouring=colours))` | R, T |
| `client.create_test_runs(manual_colouring=colours)` | colouring now goes into the protocol: `FK12(manual_colouring=colours)`; the client builds `client.test_runs` in `__init__` via `protocol.create_test_runs(graph=client.graph)` | R `client.py:176`, `protocols.py:79-175`, T |
| `from veriphix.trappifiedCanvas import TrappifiedCanvas` | gone → `veriphix.verifying.TestRun` (one per colour) | R `verifying.py:140` |
| `TrappifiedCanvas(col)` | not needed — `client.test_runs` already yields `TestRun`s | R, T |
| `run.states` (list indexed by node) | `run.input_state` — a **dict** `{node: State}` | R `verifying.py:157`, T |
| `run.traps_list` (list of `(trap,)` 1-tuples) | `run.traps` — a `frozenset[frozenset[int]]`; rebuild as `[tuple(t) for t in run.traps]` | R `verifying.py:153`, T |
| `client.delegate_test_run(backend, run, noise_model)` → per-trap bits | `client.execute_test_run(test_run, backend, noise_model, rng)` → `TestResult.trap_outcomes: dict[frozenset, int]` | R `client.py:330`, `verifying.py:217` |
| `get_bipartite_coloring(pattern)` (was a gospel import) | `veriphix.sampling_circuits.brickwork_state_transpiler.get_bipartite_coloring` (also mirrored in `veriphix.protocols`) | R `brickwork_state_transpiler.py:351`, T |

## gospel → here

| Legacy import | New location | Confirmed |
|---|---|---|
| `gospel.brickwork_state_transpiler.{ConstructionOrder, generate_random_pauli_pattern, get_bipartite_coloring}` | `veriphix.sampling_circuits.brickwork_state_transpiler` (same names) | R `brickwork_state_transpiler.py:231,376,351`, T |
| `gospel...UncorrelatedDepolarisingNoiseModel` | `veriphix.uncorrelated_depolarising_noise_model` (ported to current graphix) | new file |
| `gospel.stim_pauli_preprocessing.{StimBackend, pattern_to_stim_circuit}` | `applications/aces/stim_pauli_preprocessing.py` (ported) | new file |
| `gospel.noise_models.single_pauli_noise_model.{SinglePauliNoise, SinglePauliNoiseModel}` | `veriphix.single_pauli_noise_model` (ported to current graphix; supplied by repo owner) | new file |
| `gospel.cluster.dask_interface.get_cluster` | **dropped** (no dask) | per owner |

## graphix (changed more than the brief assumed — "verify, don't assume" paid off)

| Legacy symbol | New name / path | Confirmed |
|---|---|---|
| `graphix.noise_models.noise_model.A(noise=, nodes=)` | `graphix.noise_models.noise_model.ApplyNoise(noise=, nodes=)`; `.kind == CommandKind.ApplyNoise` | R `noise_model.py:45-68`, X `malicious_noise_model.py:9,83` |
| `graphix.noise_models.noise_model.NoiseCommands` | removed → return type is `list[CommandOrNoise]` | R `noise_model.py` |
| `graphix.noise_models.depolarising_noise_model.{DepolarisingNoise, TwoQubitDepolarisingNoise}` | `graphix.noise_models.depolarising.{...}` | R `noise_models/__init__.py` |
| `NoiseModel.input_nodes(nodes)` / `.command(cmd)` / `.confuse_result(cmd,result)` | all gained `rng=None, *, stacklevel=1` | R `noise_model.py:78-108`, X |
| `graphix.states.BasicState` (enum) + `BasicState.try_from_statevector` | **removed**. Only `graphix.states.BasicStates` (namespace of `PlanarState`) remains. Reconstructed a local `BasicState` enum in `stim_pauli_preprocessing.py` | R `states.py:81-92`, T |
| `M` command `.plane` / `.angle` | now `cmd.measurement: Measurement`; `PauliMeasurement.try_from(plane, angle)` → `BlochMeasurement(angle, plane).try_to_pauli()` (or `cmd.measurement.try_to_pauli()`) | R `command.py:M`, `measurements.py:51-132,357`, T |
| `ApplyNoise.kind == CommandKind.A` (in stim transpiler) | `CommandKind.ApplyNoise` | R `command.py:25-36`, T |
| `Pattern.nodes` / `Pattern.edges` | removed → `Pattern.extract_graph()` returns an `nx.Graph` (`.nodes` == `range(n_node)`, `.edges` are tuples) | R `client.py:142`, `pattern.py:1070`, T |
| `Pattern.get_graph()` (used by stim file) | removed → `Pattern.extract_graph()` | R, T |
| `graphix.pattern.pauli_nodes`, `graphix.sim.base_backend.BackendState` | removed; only fed `preprocess_pauli` / `StimBackendState`, which the ACES path never calls → those helpers trimmed from the vendored stim file | R, T |
| `Pattern, command, CommandKind, Statevec, DefaultMeasureMethod, PrepareMethod, State` | all still resolve | T |

### `remove_flow` behaviour change (important)
Legacy `client_pattern = remove_flow(pattern)` was fed straight to `pattern_to_stim_circuit`,
which reads each `M` command's basis. The **current** `remove_flow` strips every `M` down to
a bare `BaseM(node)` (no basis) — fine for veriphix's blind protocol (basis comes from the
`MeasureMethod`) but useless for the stim transpiler. The trap test measures **every node in
the X basis** (confirmed: `TestMeasureMethod.describe_measurement` returns `Measurement.XY(0)`
when θ-blinding is off — `client.py:425-431`). So the port derives the stim pattern from
`client.clean_pattern` and rewrites each bare `BaseM(node)` to a full `command.M(node)`
(default `Measurement.X`, empty domains). This reproduces the legacy "all-X, no feed-forward"
graph-state trap. `client.clean_pattern` already measures all `n_node` nodes (output `M`s are
added by the client) and carries no X/Z byproducts (verified at runtime).

## Invariants — status

1. **Determinism** — `PCG64(42).jumped(circuit*2 + int(order==Deviant))`: reproduced verbatim.
   The brickwork **edge** structure (the only thing `generate_equations` reads) depends solely
   on `(nqubits, nlayers, order)`, not on the random angles (R `brickwork_state_transpiler.py:268-318`),
   so `generate_edge_dependencies` is stable even though it regenerates patterns without the seed.
2. **Two orders** + deviant filter `(k%nq)%2==0 and (k//nq)%2==1`: kept verbatim.
3. **Stim default + batched sampling** `circuit.compile_sampler().sample(shots=nshots)`: kept.
   Graphix/Veriphix assert `nshots == 1`.
4. **ACES pure-Python math**: kept verbatim *except* `pattern.nodes`/`pattern.edges` →
   `pattern.extract_graph()` (graphix removed those attributes). `reversed(list(pattern))`,
   `cmd.kind == CommandKind.E`, `cmd.nodes`, `pattern.output_nodes` all still hold (T).
5. **Result-table shape** `(samples, measure_indices, traps_list)`: preserved; `traps_list`
   rebuilt as `(node,)` tuples from `run.traps`.
6. **Theory line** `lambda_expected = 1 - 4/3 * depol_prob`; plots centred on `1-4p/3` and `0`: kept.

## Behavioural differences forced by the new API
* `remove_flow` no longer keeps the measurement basis → port re-attaches X-basis `M`s (above).
* Test runs come from a `VerificationProtocol` (`FK12`) carried by the `Client`, not a
  `client.create_test_runs(...)` call; `TrappifiedCanvas` is gone.
* The single-qubit depolarising-on-each-CZ-endpoint behaviour is **not** what graphix's own
  `DepolarisingNoiseModel` does (it applies a *two-qubit* depol on the edge). The uncorrelated
  single-qubit version — which is what yields the `1 − 4p/3` per-edge eigenvalue — is the
  reason a custom model is still needed; it was ported, not replaced by the built-in.
* **Methods Stim / Graphix / Veriphix**: the three-method structure is kept.
  * **Stim** (default): fully ported and validated — the efficient batched path.
  * **Graphix** (`nshots == 1`): ported to current `simulate_pattern` + `DefaultMeasureMethod`
    + a `FixedPrepareMethod` (whose `prepare` gained the `rng` parameter); runs end-to-end.
  * **Veriphix** (`nshots == 1`): raises `NotImplementedError`. It needs a `StimBackend`
    subclass of the current `graphix.sim.base_backend.Backend`, whose ABC changed
    substantially (covariant frozen-dataclass `state` field with `init=False`, `apply_noise(cmd:
    ApplyNoise)`, a new `measure` signature, no `BackendState`). Porting it faithfully is real
    work, is not needed for the efficient path, and `client.delegate_test_run` no longer exists
    (replaced by `client.execute_test_run` returning a `TestResult.trap_outcomes` *dict*). Left
    unported on purpose — flagged, not stubbed.

## Validation results
* `python aces_experiment_veriphix.py` (Stim, defaults `nqubits=5, nlayers=10, depol_prob=0.001,
  nshots=10000, ncircuits=1`): runs end-to-end in ~12 s, writes `plot.png` + `plot_diff.png`,
  **240 edges**, inferred λ̂ **mean ≈ 0.99865** vs theory `1 − 4/3·0.001 = 0.998667`
  (|Δ| ≈ 1e-5 — well inside "a few ×10⁻³"). `plot_diff.png` is centred on 0 (mean −0.00,
  median 0.00, std 0.004).
* `python aces_experiment_veriphix.py --smoke` (`nqubits=2, nlayers=3, nshots=4000`): ~1 s,
  λ̂ mean ≈ 0.9985, for cheap re-validation of the port.
* Sanity check of the stim circuit: at `p=0` every trap node reads 0 deterministically (so the
  eigenstate preparation is correct and every single-qubit-trap stabiliser has sign +1, which is
  why `compute_failure_probabilities` can use raw outcomes without sign bookkeeping); at `p=0.1`
  the trap failure rate is clearly non-zero.

## Open questions
* ~~`SinglePauliNoise`~~ — **resolved**. The repo owner supplied the model; it was ported to
  the current graphix API as `veriphix.single_pauli_noise_model` (`Noise.nqubits` is now a
  property; `A` → `ApplyNoise`; `KrausData(coef, operator)` / `Ops.X|Z` confirmed), and the
  `SinglePauliNoise` arm of `pattern_to_stim_circuit` (`X_ERROR` / `Z_ERROR`) was reinstated.
  Not used by the depolarising ACES run, but the transpiler is now complete.

_No remaining open questions._
