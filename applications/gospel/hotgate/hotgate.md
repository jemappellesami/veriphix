# Hotgate — FK12 trap failure heatmap

## What it does

Hotgate runs VBQC verification test rounds on a family of random BQP circuits
and produces a **per-qubit failure heatmap** on the shared brickwork graph.
The goal is to identify which qubits in the resource state are most sensitive
to noise under the FK12 verification protocol.

## Protocol

Circuits are transpiled to a **brickwork state** (a fixed 2D graph whose
structure depends only on `n` and `depth`, not on the circuit content).
The FK12 protocol with **bipartite coloring** is used: the brickwork graph is
2-colorable, so each test round tests all qubits of one color as single-qubit
traps.  Each trap qubit either passes (outcome 0) or fires (outcome 1).

By aggregating outcomes over many circuits and many test rounds, the failure
rate per qubit reveals which positions in the brickwork are most exposed to
entanglement noise.

## Noise model

Depolarising noise applied at every two-qubit gate (`entanglement_error_prob`).
Single-qubit and measurement errors are set to zero to isolate the effect of
entangling operations.

## File structure

```
hotgate/
├── simulate.py   one circuit → one CSV  (SLURM worker or local)
├── plot.py       aggregate CSVs → heatmap PDF
├── hotgate.py    local convenience runner (sequential loop + auto-plot)
├── submit.sh     SLURM array job (6 noise levels × 100 circuits = 600 tasks)
└── results/
    ├── p0.1/circuit_000.csv
    ├── p0.1/circuit_001.csv
    ...
```

Each CSV has columns: `node, col, row, failure_count, total_tests`.
`col` and `row` are the qubit's position in the brickwork grid.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-qubits` | Number of logical qubits | 3 |
| `--depth` | Circuit depth | 6 |
| `--bqp-error` | BQP error folder tag | `1e-1` |
| `--n-test-rounds` | Test rounds per circuit | 100 |
| `--p-ent` | Depolarising entanglement error | `2e-3` |
| `--base-seed` | RNG base seed (task seed = base + circuit_idx) | 42 |

## Running

### On a SLURM cluster

```bash
# Submit 600 tasks (6 noise levels × 100 circuits) in parallel
sbatch applications/gospel/hotgate/submit.sh

# Check progress
find applications/gospel/hotgate/results -name "circuit_*.csv" | wc -l

# Plot once done
for p in 0.1 0.01 0.02 0.05 0.07 0.005; do
    python applications/gospel/hotgate/plot.py \
        --n-qubits 5 --depth 5 --p-ent $p
done
```

### Locally (sequential, Ctrl+C to cancel)

```bash
for p in 0.1 0.01 0.02 0.05 0.07 0.005; do
    python applications/gospel/hotgate/hotgate.py \
        --n-qubits 5 --depth 5 --p-ent $p \
        --out-dir applications/gospel/hotgate/results/p${p} \
        --out-plot applications/gospel/hotgate/heatmap_p${p}.pdf
done
```

### Resume after interruption

Both runners skip circuits whose CSV already exists.
Resubmitting `sbatch` or re-running `hotgate.py` is safe — it picks up where
it left off.  Use `--force` to rerun from scratch.

## Output

One PDF heatmap per noise level: `heatmap_p{p_ent}.pdf`.
Circles represent brickwork qubits; color encodes failure rate (YlOrRd
colormap, 0 → 95th percentile of observed rates).
Edges show the brickwork graph connectivity.

---

## For LLMs — key facts about the codebase

- **Entry point**: `simulate.py::main()` — one invocation = one circuit.
- **Pattern loading**: `read_qasm` → `transpile` → `pattern.minimize_space()`.
  All circuits of the same `(n, depth)` produce the same brickwork graph; only
  measurement angles differ.
- **FK12 instantiation**: `get_bipartite_coloring(pattern)` returns
  `(red: set[int], blue: set[int])` based on `(col + row) % 2`.
  Pass as `FK12(manual_colouring=(red, blue))`.
- **Client**: `Client(pattern, secrets, protocol, parameters, rng)` builds the
  blind patterns and test runs automatically (`autogen=True`).
- **Test run execution**: `client.sample_canvas()` → `client.delegate_canvas()`
  returns `dict[int, RunResult]`.  Filter for `TestResult` instances.
- **Trap outcome extraction**: `test_result.trap_outcomes` is
  `dict[frozenset[int], int]`.  For FK12 single-qubit traps each key is
  `frozenset({node})`; value is 0 (pass) or 1 (fire).
- **Node positions**: `get_node_positions(pattern)` returns
  `dict[int, array[int]]` where `pos = [node // n_qubits, node % n_qubits]`
  i.e. `[col, row]`.
- **CSV schema**: `node, col, row, failure_count, total_tests` — one row per
  brickwork qubit per circuit.  `plot.py` sums these across circuits.
- **Noise model**: `DepolarisingNoiseModel(entanglement_error_prob=p_ent)` with
  all other error probabilities set to 0.
- **Backend**: `DensityMatrixBackend` (required for mixed-state noise simulation).
