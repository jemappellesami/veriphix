# TODO — fast FK12 stabilizer for the Stim benchmark

## Problem

`benchmark_stim.py` / `benchmark_stim_dask.py` spend the bulk of their **build** time inside
`veriphix.verifying.build_stabilizer`, called once per FK12 test run from
`TestRun.__init__` (which runs when `Client(...)` is constructed, via
`FK12.create_test_runs`).

Measured scaling of `build` on the (width, depth) sweep is ~`|V|^2.9` (|V| = node count
= `width·(4·depth+1)`). The cost is **not** the coloring (that is the trivial O(|V|)
`get_bipartite_coloring`) — it is `build_stabilizer`:

| step | cost |
|------|------|
| `get_graph_clifford_structure` → `stim.Circuit.to_tableau()` builds a **dense Tableau on all `|V|` qubits** | O(|V|²) memory, ~O(|V|^2.5+) time |
| `clifford_structure.inverse()` | ~O(|V|^2.5) |
| conjugate ~`|V|/2` length-`|V|` Pauli strings | O(|V|³) |
| `merge(...)` Python double loop | O(|V|²) |

Extrapolated to (100, 100) → |V| ≈ 40,100 this is **days** per cell, and the dense
tableau alone is hundreds of MB. The Stim sampling itself is only ~10 min/cell (linear
in |V|), so the stabilizer build is the sole blocker to large sizes.

## Key fact (verified empirically, 7 cases up to |V|=91)

For an FK12 test run — **single-qubit X-basis traps**, one colour of a **bipartite**
(independent-set) coloring — the merged stabilizer has a closed form:

```
stabilizer letters:
  node in this colour          -> X
  opposite-colour node, deg>=1 -> Z      (every opposite-colour node, since merge
  isolated node                -> I       overlays supports and never cancels Z·Z)
sign: +1
```

This holds because:
* each single-qubit X-trap at `v` conjugates to `K_v = X_v · Π_{w∈N(v)} Z_w`
  (graph-state stabilizer generator, sign +1);
* `merge` (verifying.py) is a **support overlay**, not Pauli multiplication — so an
  opposite-colour node touched by ≥1 trap gets `Z` regardless of how many traps touch
  it (no `Z·Z = I` cancellation, hence **no degree-parity dependence**);
* same-colour nodes are an independent set, so they never receive a `Z`.

Because this reproduces the exact same `stim.PauliString`, passing it through the
existing `generate_eigenstate` reproduces `input_state` identically. Cost: **O(|V|+|E|)**
(node degrees + colour membership) — no tableau, no inverse, no conjugation, no merge.

Verification harness (already run, all match): build the closed-form PauliString from
`client.graph` + `get_bipartite_coloring`, assert `== run.stabilizer` for both colours.

## Plan

### Part A — benchmark-stim only (no veriphix edit; makes the cluster sweep feasible now) — ✅ DONE

Implemented in `benchmark_stim_dask.py` (`fk12_fast_runs` + `_FastRun`; `Cell.execute` uses
`autogen=False` + `preprocess_pattern` + `create_blind_patterns`, then the closed-form runs).
Verified: stabilizer + per-node `input_state` identical to the real `Client.test_runs` (5
sizes), and end-to-end `p_failed_round` matches the real path within Monte-Carlo noise
(0.5232 vs 0.5236 at w=9,d=16). Build step: ~2.9s → ~0.01s (≈290×) at |V|=585; speedup
grows with size (206× @ |V|=520, 640× @ |V|=979) since real is ~cubic and the fast path is
linear. Sampling is now the binding cost. Original design below:


In `benchmark_stim_dask.py` (and optionally `benchmark_stim.py`), bypass the expensive
`Client` construction inside `Cell.execute`. Today it does
`client = Client(... FK12 ...)` then uses only `client.clean_pattern` and
`client.test_runs`; the `Client` constructor is what triggers `create_test_runs` →
`build_stabilizer`. Replace with direct, cheap calls:

```python
from veriphix.client import remove_flow
from veriphix.verifying import generate_eigenstate
import stim

def fk12_fast_runs(graph, red, blue):
    """Lightweight test runs for FK12 X-basis single-qubit traps. O(|V|+|E|).

    Returns objects exposing exactly what _round_fail_pool uses:
    .input_state (dict node->PlanarState), .stabilizer (has .sign), .traps.
    """
    deg = dict(graph.degree())
    nodes = list(graph.nodes)
    runs = []
    for colour, other in ((red, blue), (blue, red)):
        letters = []
        for n in nodes:
            if n in colour:        letters.append("X")
            elif deg[n] >= 1:      letters.append("Z")   # n in `other`
            else:                  letters.append("I")
        stab = stim.PauliString("".join(letters))        # sign +1
        input_state = dict(zip(nodes, generate_eigenstate(stab)))
        traps = frozenset(frozenset([n]) for n in colour)
        runs.append(SimpleNamespace(input_state=input_state, stabilizer=stab, traps=traps))
    return runs
```

Then in `Cell.execute`:
```python
clean_pattern = remove_flow(pattern)                 # instead of client.clean_pattern
graph = pattern.extract_graph()
red, blue = get_bipartite_coloring(pattern)
test_runs = fk12_fast_runs(graph, red, blue)         # instead of client.test_runs
```
Everything downstream (`x_basis_measurement_pattern`, `_round_fail_pool`) is unchanged.

**Correctness gate before trusting any large run:** for small (width, depth) assert the
fast runs reproduce the real ones — same `stabilizer`, same per-node `input_state`,
same `p_failed_round` as the current `benchmark_stim.py`. (Cross-check already passing
for the stabilizer; extend to `input_state` and an end-to-end CSV diff.)

### Part B — veriphix (PROPOSAL — do **not** edit directly yet)

Push the fast path into the library so every FK12 user benefits:

* Option 1: a specialized `build_stabilizer_fk12(graph, colour, other_colour)` in
  `verifying.py` returning the closed-form `PauliString` directly.
* Option 2: a fast branch inside `FK12.create_test_runs` that, given the bipartite
  manual colouring and single-qubit X traps, constructs each `TestRun`'s stabilizer +
  input_state via the closed form instead of calling `build_stabilizer`. Would need a
  way to set `TestRun.stabilizer`/`input_state` without the `__init__` build (e.g. an
  alternate constructor / classmethod), since today `TestRun.__init__` unconditionally
  calls `build_stabilizer`.

Guard the fast path with the preconditions it relies on (uniform X basis, single-qubit
traps, colour is an independent set) and fall back to the generic `build_stabilizer`
otherwise. Keep the generic path as the correctness oracle in tests.

## Expected payoff

`build` drops from ~O(|V|^2.9) to ~O(|V|). The Stim sampling (~linear, ~10 min/cell at
|V|≈40k) becomes the binding constraint, so (width, depth) ≈ (100, 100) per cell moves
from "days" into "minutes" range, and the Dask sweep parallelizes those cells across the
cluster as before.
