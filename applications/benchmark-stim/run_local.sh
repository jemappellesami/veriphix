#!/bin/bash
# Local laptop run (no SLURM): LocalCluster on all cores. Omitting --walltime/--memory/
# --cores is what makes benchmark_stim_dask.py use LocalCluster instead of SLURMCluster.
#
# Usage: run_local.sh [WMIN] [WMAX] [DMIN] [DMAX] [ROUNDS] [ENT_ERRORS]
# Defaults: WMIN=2 WMAX=6 DMIN=2 DMAX=6 ROUNDS=2000 ENT_ERRORS=1e-3
#
# Keep this SMALL -- it is a wiring/sanity probe, not a data run. |V| = width*(4*depth+1),
# and per-cell cost is ~linear in |V| x rounds.
#
# Output: benchmark_stim_results_p<p_ent>_r<rounds>.csv (one per noise level, resume per file).

set -e
WMIN=${1:-2}; WMAX=${2:-6}; DMIN=${3:-2}; DMAX=${4:-6}; ROUNDS=${5:-2000}; ENT_ERRORS=${6:-1e-3}
WIDTHS=$(seq -s, "$WMIN" "$WMAX"); DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim"

echo "LOCAL stim grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX} | rounds=${ROUNDS} | p=${ENT_ERRORS}"
export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim/benchmark_stim_dask.py \
  --widths "$WIDTHS" --depths "$DEPTHS" --ent-errors "$ENT_ERRORS" \
  --rounds "$ROUNDS" --test-rounds 100 --threshold 0 \
  --out-dir "$OUT_DIR"
echo "Local run complete. Results in ${OUT_DIR}/"

# -- heatmap ----------------------------------------------------------------------
# Plot exactly the files this run wrote: one CSV per noise level, named by (p, rounds).
# `printf %.1e` reproduces Python's f"{p:.1e}" so the names match byte for byte.
# PLOT=0 skips this step; a plotting failure is reported but never discards the CSVs.
if [ "${PLOT:-1}" != "0" ]; then
  CSVS=""
  for P in ${ENT_ERRORS//,/ }; do
    TAG=$(printf "%.1e" "$P")
    CSVS="${CSVS:+$CSVS,}${OUT_DIR}/benchmark_stim_results_p${TAG}_r${ROUNDS}.csv"
  done
  echo "Plotting heatmaps for: ${CSVS}"
  ./.venv/bin/python applications/benchmark-stim/plot_stim_heatmap.py --csv "$CSVS" \
    || echo "!! heatmap generation failed -- the CSVs are still in ${OUT_DIR}"
fi
