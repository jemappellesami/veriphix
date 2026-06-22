#!/bin/bash
# Parametrized (n, t) grid run on the cluster for the Clifford+MSI honest-failure
# benchmark, sweeping one or more depolarising noise levels.
#
# Usage:
#   run_full_cluster.sh [NMIN] [NMAX] [TMIN] [TMAX] [NSHOTS] [DEPOLS]
#
# Defaults: NMIN=2 NMAX=20 TMIN=1 TMAX=20 NSHOTS=100 DEPOLS=1e-3
#
# Examples:
#   bash run_full_cluster.sh                                  # n=2..20 x t=1..20, p=1e-3
#   bash run_full_cluster.sh 2 12 1 12                        # n=2..12 x t=1..12
#   bash run_full_cluster.sh 2 20 1 20 500                    # 500 shots
#   bash run_full_cluster.sh 2 20 1 20 100 5e-4,1e-3,2e-3     # sweep 3 noise levels
#
# Output: ONE CSV per noise level, named by (p_depol, shots):
#   benchmark_msi_results_p<p_depol>_s<shots>.csv
# Runs over different (n, t) at the SAME (p_depol, shots) append to the same file;
# changing p_depol or shots writes a different file. Resume is automatic per file.

set -e

NMIN=${1:-2}
NMAX=${2:-20}
TMIN=${3:-1}
TMAX=${4:-20}
NSHOTS=${5:-100}
DEPOLS=${6:-1e-3}
TEST_ROUNDS=100

NS=$(seq -s, "$NMIN" "$NMAX")
TS=$(seq -s, "$TMIN" "$TMAX")
OUT_DIR="applications/benchmark-stim-msi"

echo "Grid: n ${NMIN}..${NMAX} x t ${TMIN}..${TMAX}  | shots=${NSHOTS} test_rounds=${TEST_ROUNDS}"
echo "Noise levels: ${DEPOLS}  | one CSV per level in ${OUT_DIR}"

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi/benchmark_msi_dask.py \
  --ns "$NS" \
  --ts "$TS" \
  --depols "$DEPOLS" \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-dir "$OUT_DIR"
