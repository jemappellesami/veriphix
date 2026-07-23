#!/bin/bash
# Parametrized grid run on the cluster, sweeping one or more noise levels, then plotting.
#
# Usage:
#   run_full_cluster.sh [WMIN] [WMAX] [DMIN] [DMAX] [ROUNDS] [ENT_ERRORS]
#
# Defaults: WMIN=1 WMAX=40 DMIN=1 DMAX=40 ROUNDS=10000 ENT_ERRORS=1e-5
#
# Examples:
#   bash run_full_cluster.sh                                    # 40x40, 10k rounds, p=1e-5
#   bash run_full_cluster.sh 1 20 1 20                          # 20x20, 10k rounds, p=1e-5
#   bash run_full_cluster.sh 1 40 1 40 100000                   # 40x40, 100k rounds
#   bash run_full_cluster.sh 1 40 1 40 10000 1.5e-5,2e-5,3e-5   # sweep 3 noise levels
#
# ROUNDS is the total honest test rounds sampled per cell -- a single flat pool, no
# shots x test_rounds grouping. It buys precision on p_failed_round directly:
# SE = sqrt(p(1-p)/ROUNDS), so 10k resolves ~1e-3 and 1e6 resolves ~1e-5. p_false_reject is
# derived analytically as P[Binom(--test-rounds, p_failed_round) > --threshold], and
# n_fail/n_rounds go into the CSV so any (R, w) can be re-derived without re-simulating.
#
# Output: ONE CSV per noise level, named by (p_ent, rounds):
#   benchmark_stim_results_p<p_ent>_r<rounds>.csv
# Runs over different widths/depths at the SAME (p_ent, rounds) append to the same file;
# changing p_ent or rounds writes to a different file. Resume is automatic per file.

set -e

WMIN=${1:-1}
WMAX=${2:-40}
DMIN=${3:-1}
DMAX=${4:-40}
ROUNDS=${5:-10000}
ENT_ERRORS=${6:-1e-5}

WIDTHS=$(seq -s, "$WMIN" "$WMAX")
DEPTHS=$(seq -s, "$DMIN" "$DMAX")
OUT_DIR="applications/benchmark-stim"

echo "Grid: width ${WMIN}..${WMAX} x depth ${DMIN}..${DMAX}  | rounds=${ROUNDS}/cell"
echo "Noise levels: ${ENT_ERRORS}  | one CSV per level in ${OUT_DIR}"

export PYTHONUNBUFFERED=1
./.venv/bin/python -u applications/benchmark-stim/benchmark_stim_dask.py \
  --widths "$WIDTHS" \
  --depths "$DEPTHS" \
  --ent-errors "$ENT_ERRORS" \
  --rounds "$ROUNDS" --test-rounds 100 --threshold 0 \
  --walltime 4 --memory 8 --cores 4 --port 8787 --scale 40 \
  --out-dir "$OUT_DIR"

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
