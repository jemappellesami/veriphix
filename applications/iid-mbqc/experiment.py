#!/usr/bin/env python3
"""MBQC benchmarking stage: measure the test-round failure count per (width, depth) tile.

One tile = one brickwork graph state of the given (width, depth), one noise level, and
``--rounds`` independent test rounds. Each round draws one of the two FK12 test runs
uniformly and passes or fails; the tile reports the integer pair ``(n_fail, n_rounds)``.

Integers, not a rate: Clopper-Pearson is a function of the count, and reconstructing Y by
rounding a stored rate loses the exactness the bound is chosen for. Downstream everything
is derived by ``iid_pipeline.py``, so this script stores no derived quantity except
``p_failed_round`` for eyeballing.

The per-run split (``n_fail_run0``/``n_fail_run1``) is free -- the rounds are already
sampled per test run -- and records how unequal the two runs' detection rates are. The
pipeline's ``r <= k q`` step is worst-case over which run catches a harmful error; if the
two columns come out close on real data, that worst case is loose and there is a factor to
reclaim.

Scaling
-------
Cost per tile is dominated by two ``compile_sampler`` calls plus ``rounds`` Clifford shots,
so it grows with the graph size (~width*depth nodes) and linearly in ``--rounds``. The
laptop defaults below are a concept check: a few small tiles, a few thousand rounds, a few
seconds. For the real picture raise ``--widths/--depths/--rounds`` on the cluster.

Re-runnable: results append to the CSV and existing ``(p_ent, width, depth)`` rows are
skipped, so an interrupted sweep resumes and a grid can be widened without recomputing.

Usage:
    python applications/iid-mbqc/experiment.py --smoke
    python applications/iid-mbqc/experiment.py --widths 2,3,4 --depths 2,4,6 --rounds 4000
    python applications/iid-mbqc/experiment.py --widths "2,3,4,5,6,7,8,9,10" --depths "2,3,4,5,6,7,8,9,10" --rounds 4000
    python applications/iid-mbqc/experiment.py --widths 2,4,6,8 --depths 2,4,8,16 \
        --p-ent 1e-3,1e-2 --rounds 100000 --out results/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from graphix import Pattern, command
from graphix.command import CommandKind
from graphix.sim.statevec import Statevec
from numpy.random import PCG64, Generator

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "applications" / "aces"))  # vendored stim transpiler

from stim_pauli_preprocessing import BasicState, pattern_to_stim_circuit  # noqa: E402

from veriphix.blinding import Secrets  # noqa: E402
from veriphix.client import Client  # noqa: E402
from veriphix.protocols import FK12  # noqa: E402
from veriphix.sampling_circuits.brickwork_state_transpiler import (  # noqa: E402
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from veriphix.uncorrelated_depolarising_noise_model import (  # noqa: E402
    UncorrelatedDepolarisingNoiseModel,
)

CSV_FIELDS = [
    "p_ent",
    "width",
    "depth",
    "n_fail",
    "n_rounds",
    "p_failed_round",
    "n_fail_run0",
    "n_rounds_run0",
    "n_fail_run1",
    "n_rounds_run1",
    "nodes",
    "elapsed_s",
]

# Laptop-sized defaults -- a concept check, not the real sweep.
DEFAULT_WIDTHS = "2,3,4"
DEFAULT_DEPTHS = "2,4,6"
DEFAULT_P_ENT = "1e-3"
DEFAULT_ROUNDS = 4000
BASE_SEED = 12345


def _state_to_basic_state(state: object) -> BasicState:
    bs = BasicState.try_from_statevector(Statevec(state).psi)
    if bs is None:
        raise ValueError(f"Not a basic state: {state}")
    return bs


def _x_basis_measurement_pattern(clean_pattern: Pattern) -> Pattern:
    """Re-attach an X-basis ``M`` to every bare ``BaseM`` left by flow removal."""
    pattern = Pattern(input_nodes=clean_pattern.input_nodes)
    for cmd in clean_pattern:
        if cmd.kind == CommandKind.M:
            pattern.add(command.M(node=cmd.node))
        else:
            pattern.add(cmd)
    return pattern


def _fail_samples(run: object, stim_pattern: Pattern, noise_model: object, shots: int) -> np.ndarray:
    """Boolean ``(shots,)``: which rounds of this one test run failed.

    A round fails if any trap's parity, XORed with the stabiliser sign, is 1.
    """
    if shots == 0:
        return np.zeros(0, dtype=bool)
    input_state: dict[int, BasicState] = {}
    fixed_states: dict[int, BasicState] = {}
    for node, state in run.input_state.items():
        bs = _state_to_basic_state(state)
        target = input_state if node in stim_pattern.input_nodes else fixed_states
        target[node] = bs
    circuit, measure_indices = pattern_to_stim_circuit(
        stim_pattern, input_state=input_state, noise_model=noise_model, fixed_states=fixed_states
    )
    samples = np.asarray(circuit.compile_sampler().sample(shots=shots))
    sign_flip = int(run.stabilizer.sign == -1)
    failed = np.zeros(shots, dtype=bool)
    for trap in run.traps:
        cols = [measure_indices[node] for node in trap]
        failed |= ((samples[:, cols].sum(axis=1) & 1) ^ sign_flip).astype(bool)
    return failed


def simulate_tile(width: int, depth: int, p_ent: float, rounds: int, base_seed: int) -> dict:
    """One (width, depth, p_ent) tile -> a CSV row of raw counts."""
    t0 = time.monotonic()
    # Deterministic per (width, depth), so the same graph is reused across noise levels.
    rng = Generator(PCG64(base_seed).jumped(width * 1009 + depth))
    pattern = generate_random_pauli_pattern(
        nqubits=width, nlayers=depth, order=ConstructionOrder.Canonical, rng=rng
    )
    client = Client(
        pattern=pattern,
        secrets=Secrets(r=False, a=False, theta=False),
        protocol=FK12(manual_colouring=list(get_bipartite_coloring(pattern))),
        rng=rng,
    )
    stim_pattern = _x_basis_measurement_pattern(client.clean_pattern)
    test_runs = client.test_runs
    if len(test_runs) != 2:
        raise RuntimeError(f"expected 2 FK12 test runs, got {len(test_runs)}")

    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=p_ent)
    # FK12 picks a test run uniformly per round. Draw the split first, then sample each run
    # exactly as often as it was drawn: two compiles per tile and no discarded shots.
    counts = np.bincount(rng.integers(0, 2, size=rounds), minlength=2)
    per_run = [int(_fail_samples(run, stim_pattern, noise_model, int(n)).sum()) for run, n in zip(test_runs, counts, strict=True)]

    n_fail = sum(per_run)
    return {
        "p_ent": p_ent,
        "width": width,
        "depth": depth,
        "n_fail": n_fail,
        "n_rounds": rounds,
        "p_failed_round": n_fail / rounds,
        "n_fail_run0": per_run[0],
        "n_rounds_run0": int(counts[0]),
        "n_fail_run1": per_run[1],
        "n_rounds_run1": int(counts[1]),
        "nodes": pattern.n_node,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _load_done(path: Path) -> tuple[set[tuple[str, str, str]], set[int]]:
    """Already-computed ``(p_ent, width, depth)`` keys, plus the ``s`` values in the file.

    The key deliberately excludes ``n_rounds`` so that resuming an interrupted sweep does not
    re-measure finished tiles. The cost is that widening a grid with a *different*
    ``--rounds`` silently mixes two values of ``s`` in one file, which downstream cannot
    undo: ``q_U`` depends on ``s``, so an under-measured tile looks noisier than it is. The
    caller compares the returned set against the requested ``rounds`` and warns.
    """
    if not path.exists():
        return set(), set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(r["p_ent"], r["width"], r["depth"]) for r in rows}
    seen_s = {int(r["n_rounds"]) for r in rows if r.get("n_rounds")}
    return keys, seen_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--widths", default=DEFAULT_WIDTHS)
    parser.add_argument("--depths", default=DEFAULT_DEPTHS)
    parser.add_argument("--p-ent", default=DEFAULT_P_ENT, help="comma-separated entanglement error rates")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="s, test rounds per tile")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "mbqc_iid.csv")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--smoke", action="store_true", help="tiny 2x2 grid, 500 rounds")
    args = parser.parse_args()

    if args.smoke:
        widths, depths, p_ents, rounds = [2, 3], [2, 4], [1e-3], 500
    else:
        widths, depths, p_ents = _parse_ints(args.widths), _parse_ints(args.depths), _parse_floats(args.p_ent)
        rounds = args.rounds

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, seen_s = _load_done(args.out)
    cells = [(p, w, d) for p in p_ents for w in widths for d in depths]
    todo = [c for c in cells if (repr(c[0]), str(c[1]), str(c[2])) not in done]

    print(f"widths={widths} depths={depths} p_ent={p_ents} rounds={rounds}")
    print(f"{len(cells)} tiles, {len(cells) - len(todo)} already in {args.out}, {len(todo)} to run")

    if todo and seen_s - {rounds}:
        print(
            f"\n!! {args.out.name} already holds tiles at s={sorted(seen_s)}, and this run adds "
            f"s={rounds}.\n"
            f"   q_U depends on s, so tiles measured with different s are not comparable: an\n"
            f"   under-measured tile gets a wider interval and looks noisier than it is. The\n"
            f"   heatmaps and the volume frontier would mix noise with measurement effort.\n"
            f"   Write to a different --out, or re-measure the whole grid at one s.\n"
        )

    write_header = not args.out.exists()
    t_start = time.monotonic()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for i, (p_ent, width, depth) in enumerate(todo, 1):
            try:
                row = simulate_tile(width, depth, p_ent, rounds, args.seed)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] w={width} d={depth} p={p_ent:.1e} FAILED: {exc}")
                continue
            writer.writerow(row)
            handle.flush()
            print(
                f"[{i}/{len(todo)}] w={width} d={depth} p={p_ent:.1e} "
                f"nodes={row['nodes']} q_hat={row['p_failed_round']:.4f} "
                f"({row['n_fail']}/{row['n_rounds']})  runs=({row['n_fail_run0']}/{row['n_rounds_run0']}, "
                f"{row['n_fail_run1']}/{row['n_rounds_run1']})  {row['elapsed_s']}s"
            )

    print(f"done in {time.monotonic() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
