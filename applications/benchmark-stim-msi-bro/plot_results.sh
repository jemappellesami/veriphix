#!/usr/bin/env bash
# Generate discrete heatmaps (p_failed_round with feasibility frontiers, and p_false_reject)
# for every p_depol present in the Broadbent Clifford+MSI results CSV(s). Mirrors
# applications/benchmark-stim/plot_results.sh.
#
# Usage:
#   applications/benchmark-stim-msi-bro/plot_results.sh [csv] [outdir]
# csv defaults to ALL benchmark_bro_results_p*_s*.csv in this folder; outdir defaults to
# applications/benchmark-stim-msi-bro/heatmaps.
set -euo pipefail

HERE="applications/benchmark-stim-msi-bro"
CSV="${1:-}"
OUTDIR="${2:-$HERE/heatmaps}"

CSV_ARG=()
if [[ -n "$CSV" ]]; then
    CSV_ARG=(--csv "$CSV")
fi

for metric in p_failed_round p_false_reject; do
    ./.venv/bin/python "$HERE/plot_bro_heatmap.py" \
        ${CSV_ARG[@]+"${CSV_ARG[@]}"} \
        --metric "$metric" \
        --outdir "$OUTDIR"
done
