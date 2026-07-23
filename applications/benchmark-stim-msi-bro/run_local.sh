#!/bin/bash
# Local run (no SLURM): LocalCluster on all cores. Omitting --walltime/--memory/--cores is
# what makes benchmark_bro_dask.py use LocalCluster instead of SLURMCluster.
#
# Usage: run_local.sh [WMIN] [WMAX] [DMIN] [DMAX] [ROUNDS] [DEPOLS]
# Defaults: WMIN=2 WMAX=6 DMIN=2 DMAX=6 ROUNDS=2000 DEPOLS=1e-3
#
# Keep this SMALL -- it is a wiring/sanity probe, not a data run.

set -e
WMIN=${1:-2}; WMAX=${2:-6}; DMIN=${3:-2}; DMAX=${4:-6}; ROUNDS=${5:-2000}; DEPOLS=${6:-1e-3}
WIDTHS=$(seq -s, "$WMIN" "$WMAX"); DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim-msi-bro"

echo "LOCAL bro grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX} | rounds=${ROUNDS} | p=${DEPOLS}"
export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim-msi-bro/benchmark_bro_dask.py \
  --widths "$WIDTHS" --depths "$DEPTHS" --depols "$DEPOLS" \
  --rounds "$ROUNDS" --test-rounds 100 --threshold 0 --out-dir "$OUT_DIR"
echo "Local run complete. Results in ${OUT_DIR}/"

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
