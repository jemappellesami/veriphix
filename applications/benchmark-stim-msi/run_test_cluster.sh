#!/bin/bash
# Small test run on the cluster (5x5 (n,t) grid = 25 cells, ~1 min)

set -e

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi/benchmark_msi_dask.py \
  --ns 2,3,4,5,6 \
  --ts 1,2,3,4,5 \
  --depols 1e-3 \
  --shots 100 --test-rounds 100 \
  --walltime 1 --memory 8 --cores 4 --port 8787 --scale 4 \
  --out-dir applications/benchmark-stim-msi

echo "Test run complete. Results in applications/benchmark-stim-msi/"
