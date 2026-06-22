#!/bin/bash
# Small test run on the cluster (5x5 (width,depth) grid = 25 cells, ~1 min) to check the
# SLURM wiring before launching a big sweep.

set -e

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \
  --widths 2,3,4,5,6 \
  --depths 1,2,3,4,5 \
  --depols 1e-3 \
  --shots 100 --test-rounds 100 \
  --walltime 1 --memory 8 --cores 4 --port 8787 --scale 4 \
  --out-dir applications/benchmark-stim-msi-bro

echo "Test run complete. Results in applications/benchmark-stim-msi-bro/"
