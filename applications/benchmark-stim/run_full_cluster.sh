#!/bin/bash
# Parametrized grid run on the cluster, noise prob 1e-5.
#
# Usage:
#   run_full_cluster.sh [WMIN] [WMAX] [DMIN] [DMAX] [NSHOTS]
#
# Defaults: WMIN=1 WMAX=40 DMIN=1 DMAX=40 NSHOTS=100  (the full 40x40 grid).
#
# Examples:
#   bash run_full_cluster.sh                 # 1..40 x 1..40, 100 shots
#   bash run_full_cluster.sh 1 20 1 20       # 1..20 x 1..20, 100 shots
#   bash run_full_cluster.sh 1 40 1 40 500   # 1..40 x 1..40, 500 shots
#
# The output CSV is named by NSHOTS only (the precision), NOT by the dimension range.
# So runs over different widths/depths at the SAME shot count all append to the same
# file; changing NSHOTS writes to a different file. Resume is automatic: re-running
# skips any (p_ent, width, depth) tile already in that CSV.

set -e
cd /Users/sabdulsa/Codes/veriphix

WMIN=${1:-1}
WMAX=${2:-40}
DMIN=${3:-1}
DMAX=${4:-40}
NSHOTS=${5:-100}
TEST_ROUNDS=100

WIDTHS=$(seq -s, "$WMIN" "$WMAX")
DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_CSV="applications/benchmark-stim/benchmark_stim_results_1e5_s${NSHOTS}.csv"

echo "Grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX}  | shots=${NSHOTS} test_rounds=${TEST_ROUNDS}"
echo "Output: ${OUT_CSV}"

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim/benchmark_stim_dask.py \
  --widths "$WIDTHS" \
  --depths "$DEPTHS" \
  --ent-errors 1e-5 \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-csv "$OUT_CSV"

echo "Done. Results in ${OUT_CSV}"
