#!/bin/bash
# Local run (no SLURM): a LocalCluster on all CPU cores. Omitting --walltime/--memory/
# --cores is what makes benchmark_msi_dask.py use LocalCluster instead of SLURMCluster
# (so it never calls `sbatch`). Good for a laptop smoke / small grids.
#
# Usage:
#   run_local.sh [NMIN] [NMAX] [TMIN] [TMAX] [NSHOTS] [DEPOLS]
# Defaults: NMIN=2 NMAX=6 TMIN=1 TMAX=5 NSHOTS=100 DEPOLS=1e-3

set -e

NMIN=${1:-2}
NMAX=${2:-6}
TMIN=${3:-1}
TMAX=${4:-5}
NSHOTS=${5:-100}
DEPOLS=${6:-1e-2}
TEST_ROUNDS=100

NS=$(seq -s, "$NMIN" "$NMAX")
TS=$(seq -s, "$TMIN" "$TMAX")
OUT_DIR="applications/benchmark-stim-msi"

echo "LOCAL grid: n ${NMIN}..${NMAX} x t ${TMIN}..${TMAX}  | shots=${NSHOTS} test_rounds=${TEST_ROUNDS} | p=${DEPOLS}"

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi/benchmark_msi_dask.py \
  --ns "$NS" \
  --ts "$TS" \
  --depols "$DEPOLS" \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" \
  --out-dir "$OUT_DIR"

echo "Local run complete. Results in ${OUT_DIR}/"
