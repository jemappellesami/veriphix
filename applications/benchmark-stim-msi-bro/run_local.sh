#!/bin/bash
# Local run (no SLURM): LocalCluster on all cores. Omitting --walltime/--memory/--cores is
# what makes benchmark_bro_dask.py use LocalCluster instead of SLURMCluster.
#
# Usage: run_local.sh [WMIN] [WMAX] [DMIN] [DMAX] [NSHOTS] [DEPOLS]
# Defaults: WMIN=2 WMAX=6 DMIN=2 DMAX=6 NSHOTS=100 DEPOLS=1e-3

set -e
WMIN=${1:-2}; WMAX=${2:-6}; DMIN=${3:-2}; DMAX=${4:-6}; NSHOTS=${5:-100}; DEPOLS=${6:-1e-3}
TEST_ROUNDS=100
WIDTHS=$(seq -s, "$WMIN" "$WMAX"); DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim-msi-bro"

echo "LOCAL bro grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX} | shots=${NSHOTS} | p=${DEPOLS}"
export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \
  --widths "$WIDTHS" --depths "$DEPTHS" --depols "$DEPOLS" \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" --out-dir "$OUT_DIR"
echo "Local run complete. Results in ${OUT_DIR}/"
