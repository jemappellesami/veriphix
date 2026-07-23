#!/bin/bash
# Small test run on the cluster (5x5 grid = 25 cells, ~2 min) to check the SLURM wiring
# before launching a big sweep.

set -e

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim/benchmark_stim_dask.py \
  --widths 1,2,3,4,5 \
  --depths 1,2,3,4,5 \
  --ent-errors 1e-5 \
  --rounds 10000 --test-rounds 100 --threshold 0 \
  --walltime 1 --memory 8 --cores 4 --port 8787 --scale 4 \
  --out-dir applications/benchmark-stim

echo "Test run complete. Results in applications/benchmark-stim/"

# -- heatmap ----------------------------------------------------------------------
# Plot the file this run wrote. PLOT=0 skips; a plotting failure never discards the CSV.
if [ "${PLOT:-1}" != "0" ]; then
  ./.venv/bin/python applications/benchmark-stim/plot_stim_heatmap.py \
    --csv applications/benchmark-stim/benchmark_stim_results_p1.0e-05_r10000.csv \
    || echo "!! heatmap generation failed -- the CSV is still in applications/benchmark-stim"
fi
