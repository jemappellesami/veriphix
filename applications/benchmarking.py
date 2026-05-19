"""Verification-based benchmark — data acquisition only.

Sweeps a grid of (width, depth) over a range of entanglement error rates,
estimates the honest false-reject probability via Monte Carlo simulation, and
appends every result row to a CSV file as it is computed.

Plotting is handled separately by applications/plot_veriphix_heatmaps.py.

Usage
-----
    python applications/benchmarking.py

Output
------
    applications/veriphix_benchmark_results.csv
    Columns: p_ent, width, depth, p_failed_round, p_false_reject
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from graphix.noise_models import DepolarisingNoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend

from veriphix.blinding import Secrets
from veriphix.client import Client
from veriphix.protocols import RandomTraps
from veriphix.verifying import TrappifiedSchemeParameters

# ── parameters ────────────────────────────────────────────────────────────────
WIDTHS      = [2, 3, 4]   # nqubits
DEPTHS      = [2, 3, 4]   # circuit depth
TEST_ROUNDS = 30
N_SHOTS     = 100

# Low end (sparse) + dense log-sweep from 1e-3 to 1e-1
ENT_ERRORS: list[float] = (
    # [1e-6, 1e-5, 1e-4]+
    [
        # 1e-3, 
    2.2e-3, 
    4.6e-3, 
    1e-2, 
    2.2e-2, 
    4.6e-2, 
    1e-1
    ]
    # list(np.logspace(-3, -1, num=7))   # 1e-3, ~2.2e-3, ~4.6e-3, 1e-2, ~2.2e-2, ~4.6e-2, 1e-1
)

OUT_CSV = Path("applications/veriphix_benchmark_results.csv")
CSV_HEADER = ["p_ent", "width", "depth", "p_failed_round", "p_false_reject"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _open_csv(path: Path) -> tuple[csv.DictWriter, object]:
    """Open the CSV for appending, writing the header only if the file is new."""
    is_new = not path.exists() or path.stat().st_size == 0
    fh = path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
    if is_new:
        writer.writeheader()
    return writer, fh


# ── sweep ─────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

total_noise_levels = len(ENT_ERRORS)
total_cells        = len(DEPTHS) * len(WIDTHS)
grand_total        = total_noise_levels * total_cells
grand_cell         = 0
t_run_start        = time.monotonic()

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
csv_writer, csv_fh = _open_csv(OUT_CSV)

try:
    for p_idx, ent_error in enumerate(ENT_ERRORS):
        noise_model = DepolarisingNoiseModel(
            entanglement_error_prob=ent_error,
            measure_error_prob=0.0,
            x_error_prob=0.0,
            z_error_prob=0.0,
            measure_channel_prob=0.0,
        )

        print(f"\n{'='*64}")
        print(f"Noise level [{p_idx + 1}/{total_noise_levels}]  p_ent = {ent_error:.2e}")
        print(f"{'='*64}")

        t_level_start = time.monotonic()

        for i, depth in enumerate(DEPTHS):
            for j, nqubits in enumerate(WIDTHS):
                cell         = i * len(WIDTHS) + j + 1
                grand_cell  += 1
                t_cell_start = time.monotonic()

                print(f"\n  [{cell}/{total_cells}] n={nqubits}, depth={depth}")

                failed_round_fractions: list[float] = []
                false_rejects: list[bool] = []

                for shot in range(N_SHOTS):
                    elapsed = time.monotonic() - t_cell_start
                    avg     = elapsed / shot if shot > 0 else 0.0
                    eta_cell = avg * (N_SHOTS - shot)
                    print(
                        f"    shot {shot + 1}/{N_SHOTS}  "
                        f"elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta_cell)}",
                        end="\r",
                        flush=True,
                    )

                    circuit = rand_circuit(nqubits, depth, rng)
                    pattern = circuit.transpile().pattern

                    client = Client(
                        pattern=pattern,
                        secrets=Secrets(a=True, r=True, theta=True),
                        protocol=RandomTraps(),
                        parameters=TrappifiedSchemeParameters(
                            comp_rounds=0, test_rounds=TEST_ROUNDS, threshold=0
                        ),
                        rng=rng,
                    )

                    canvas   = client.sample_canvas(rng=rng)
                    outcomes = client.delegate_canvas(
                        canvas=canvas,
                        backend_cls=DensityMatrixBackend,
                        noise_model=noise_model,
                        rng=rng,
                    )
                    traps_ok, _, result_analysis = client.analyze_outcomes(canvas, outcomes)

                    failed_round_fractions.append(result_analysis.nr_failed_test_rounds / TEST_ROUNDS)
                    false_rejects.append(not traps_ok)

                p_failed_round = float(np.mean(failed_round_fractions))
                p_false_reject = float(np.mean(false_rejects))

                # ── write row immediately so partial runs are recoverable ──────
                csv_writer.writerow({
                    "p_ent":          ent_error,
                    "width":          nqubits,
                    "depth":          depth,
                    "p_failed_round": p_failed_round,
                    "p_false_reject": p_false_reject,
                })
                csv_fh.flush()  # type: ignore[union-attr]

                cell_time     = time.monotonic() - t_cell_start
                level_elapsed = time.monotonic() - t_level_start
                run_elapsed   = time.monotonic() - t_run_start
                avg_grand     = run_elapsed / grand_cell
                eta_total     = avg_grand * (grand_total - grand_cell)

                print(
                    f"    → p_failed_round={p_failed_round:.4f}  "
                    f"p_false_reject={p_false_reject:.4f}  "
                    f"| cell {_fmt_time(cell_time)}"
                    f"  level {_fmt_time(level_elapsed)}"
                    f"  total {_fmt_time(run_elapsed)}"
                    f"  ETA {_fmt_time(eta_total)}"
                )

finally:
    csv_fh.close()  # type: ignore[union-attr]

print(f"\nDone. Results saved to {OUT_CSV}")
print(f"Total runtime: {_fmt_time(time.monotonic() - t_run_start)}")
