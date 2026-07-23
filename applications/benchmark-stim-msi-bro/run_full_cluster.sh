#!/bin/bash
# Parametrized (width, depth) grid run on SLURM for the Broadbent-compiled Clifford+MSI
# honest-failure benchmark (FK12-analogue bipartite traps), sweeping noise levels.
#
# Usage: run_full_cluster.sh [WMIN] [WMAX] [DMIN] [DMAX] [ROUNDS] [DEPOLS]
# Defaults: WMIN=1 WMAX=40 DMIN=1 DMAX=40 ROUNDS=10000 DEPOLS=1e-3
#
# ROUNDS is the total honest test rounds per cell -- one flat pool, no shots x test_rounds
# grouping. SE on p_failed_round = sqrt(p(1-p)/ROUNDS): 10k resolves ~1e-3, 1e6 ~1e-5.
# p_false_reject is derived analytically as P[Binom(--test-rounds, p_failed_round) > w],
# and n_fail/n_rounds land in the CSV so any (R, w) is re-derivable without re-simulating.
#
# Examples:
#   bash run_full_cluster.sh                                 # 40x40, 10k rounds, p=1e-3
#   bash run_full_cluster.sh 1 40 1 40 10000 "1e-3,1e-4,1e-5"
#
# One CSV per noise level: benchmark_bro_results_p<p_depol>_r<rounds>.csv (resume per file).
# NOTE: wire count grows ~ width*(1+2*depth) (Broadbent H-gadget overhead). Trap setup is now
# fully O(gates) (structural segment colouring + closed-form |0>/|+> prep, no tableau), so the
# binding per-cell cost is the Stim sampling; large width,depth are feasible.

set -e
WMIN=${1:-1}; WMAX=${2:-40}; DMIN=${3:-1}; DMAX=${4:-40}; ROUNDS=${5:-10000}; DEPOLS=${6:-1e-3}
WIDTHS=$(seq -s, "$WMIN" "$WMAX"); DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim-msi-bro"

echo "Grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX} | rounds=${ROUNDS}/cell"
echo "Noise levels: ${DEPOLS} | one CSV per level in ${OUT_DIR}"
export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \
  --widths "$WIDTHS" --depths "$DEPTHS" --depols "$DEPOLS" \
  --rounds "$ROUNDS" --test-rounds 100 --threshold 0 \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-dir "$OUT_DIR"

# -- heatmap ----------------------------------------------------------------------
# Plot exactly the files this run wrote: one CSV per noise level, named by (p, rounds).
# `printf %.1e` reproduces Python's f"{p:.1e}" so the names match byte for byte.
# PLOT=0 skips this step; a plotting failure is reported but never discards the CSVs.
if [ "${PLOT:-1}" != "0" ]; then
  CSVS=""
  for P in ${DEPOLS//,/ }; do
    TAG=$(printf "%.1e" "$P")
    CSVS="${CSVS:+$CSVS,}${OUT_DIR}/benchmark_bro_results_p${TAG}_r${ROUNDS}.csv"
  done
  echo "Plotting heatmaps for: ${CSVS}"
  ./.venv/bin/python applications/benchmark-stim-msi-bro/plot_bro_heatmap.py --csv "$CSVS" \
    || echo "!! heatmap generation failed -- the CSVs are still in ${OUT_DIR}"
fi
