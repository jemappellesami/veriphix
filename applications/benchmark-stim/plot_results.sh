#!/usr/bin/env bash
# Generate discrete heatmaps (p_failed_round and p_false_reject) for every p_ent
# value present in a benchmark_stim_results.csv, mirroring the plots produced
# by applications/plot_veriphix_heatmaps.py for the density-matrix benchmark.
#
# Usage:
#   applications/benchmark-stim/plot_results.sh [csv] [outdir]
set -euo pipefail

CSV="${1:-applications/benchmark-stim/benchmark_stim_results.csv}"
OUTDIR="${2:-applications/benchmark-stim/heatmaps}"

for metric in p_failed_round p_false_reject; do
    ./.venv/bin/python applications/plot_veriphix_heatmaps.py \
        --csv "$CSV" \
        --metric "$metric" \
        --outdir "$OUTDIR"
done
