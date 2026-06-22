#!/bin/bash
# Parametrized (width, depth) grid run on SLURM for the Broadbent-compiled Clifford+MSI
# honest-failure benchmark (FK12-analogue bipartite traps), sweeping noise levels.
#
# Usage: run_full_cluster.sh [WMIN] [WMAX] [DMIN] [DMAX] [NSHOTS] [DEPOLS]
# Defaults: WMIN=2 WMAX=12 DMIN=1 DMAX=12 NSHOTS=100 DEPOLS=1e-3
#
# Examples:
#   bash run_full_cluster.sh                               # 2..12 x 1..12, p=1e-3
#   bash run_full_cluster.sh 2 12 1 12 100 "1e-3,1e-4,1e-5"
#
# One CSV per noise level: benchmark_bro_results_p<p_depol>_s<shots>.csv (resume per file).
# NOTE: wire count grows ~ width*(1+2*depth) (Broadbent H-gadget overhead). Trap setup is now
# fully O(gates) (structural segment colouring + closed-form |0>/|+> prep, no tableau), so the
# binding per-cell cost is the Stim sampling; large width,depth are feasible.

set -e
WMIN=${1:-2}; WMAX=${2:-12}; DMIN=${3:-1}; DMAX=${4:-12}; NSHOTS=${5:-100}; DEPOLS=${6:-1e-3}
TEST_ROUNDS=100
WIDTHS=$(seq -s, "$WMIN" "$WMAX"); DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim-msi-bro"

echo "Grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX} | shots=${NSHOTS} test_rounds=${TEST_ROUNDS}"
echo "Noise levels: ${DEPOLS} | one CSV per level in ${OUT_DIR}"
export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \
  --widths "$WIDTHS" --depths "$DEPTHS" --depols "$DEPOLS" \
  --shots "$NSHOTS" --test-rounds "$TEST_ROUNDS" \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-dir "$OUT_DIR"
