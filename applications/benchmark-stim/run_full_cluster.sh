#!/bin/bash
# Parametrized grid run on the cluster, sweeping one or more noise levels.
#
# Usage:
#   run_full_cluster.sh [WMIN] [WMAX] [DMIN] [DMAX] [NSHOTS] [ENT_ERRORS]
#
# Defaults: WMIN=1 WMAX=40 DMIN=1 DMAX=40 NSHOTS=100 ENT_ERRORS=1e-5
#
# Examples:
#   bash run_full_cluster.sh                                   # 40x40, 100 shots, p=1e-5
#   bash run_full_cluster.sh 1 20 1 20                         # 20x20, 100 shots, p=1e-5
#   bash run_full_cluster.sh 1 40 1 40 500                     # 40x40, 500 shots, p=1e-5
#   bash run_full_cluster.sh 1 40 1 40 100 1.5e-5,2e-5,3e-5    # sweep 3 noise levels
#
# Output: ONE CSV per noise level, named by (p_ent, shots):
#   benchmark_stim_results_p<p_ent>_s<shots>.csv
# So runs over different widths/depths at the SAME (p_ent, shots) append to the same
# file; changing p_ent or shots writes to a different file. Resume is automatic per file.

set -e

WMIN=${1:-1}
WMAX=${2:-40}
DMIN=${3:-1}
DMAX=${4:-40}
NSHOTS=${5:-100}
ENT_ERRORS=${6:-1e-5}
TEST_ROUNDS=100

WIDTHS=$(seq -s, "$WMIN" "$WMAX")
DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim"

echo "Grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX}  | shots=${NSHOTS} test_rounds=${TEST_ROUNDS}"
echo "Noise levels: ${ENT_ERRORS}  | one CSV per level in ${OUT_DIR}"

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim/benchmark_stim_dask.py \
  --widths "$WIDTHS" \
  --depths "$DEPTHS" \
  --ent-errors "$ENT_ERRORS" \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-dir "$OUT_DIR"
