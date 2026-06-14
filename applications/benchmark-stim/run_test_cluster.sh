#!/bin/bash
# Small test run on the cluster (5×5 grid = 25 cells, ~2 min)

set -e
cd /Users/sabdulsa/Codes/veriphix

./.venv/bin/python applications/benchmark-stim/benchmark_stim_dask.py \
  --widths 1,2,3,4,5 \
  --depths 1,2,3,4,5 \
  --ent-errors 1e-5 \
  --shots 100 --test-rounds 100 \
  --walltime 1 --memory 8 --cores 4 --port 8787 --scale 4 \
  --out-csv applications/benchmark-stim/benchmark_stim_results_test_1e5.csv

echo "Test run complete. Results in applications/benchmark-stim/benchmark_stim_results_test_1e5.csv"
