#!/usr/bin/env bash
# Generate the p_failed_round heatmap (with feasibility frontiers) for the Broadbent
# Clifford+MSI results, optionally selecting CSVs by SHOT COUNT.
#
# Usage:
#   applications/benchmark-stim-msi-bro/plot_results.sh [SHOTS] [OUTDIR]
#
#   SHOTS   shot count to plot (e.g. 100, 1000, 10000). Default: all shot counts.
#   OUTDIR  output dir. Default: applications/benchmark-stim-msi-bro/heatmaps
#
# Examples:
#   plot_results.sh                # every benchmark_bro_results_p*_s*.csv
#   plot_results.sh 1000           # only the s1000 files (one figure per noise level)
#   plot_results.sh 10000 /tmp/hm  # s10000 files, custom output dir
#
# One figure per (noise level, metric). Picks up all noise levels present for that shot count.
set -euo pipefail

HERE="applications/benchmark-stim-msi-bro"
SHOTS="${1:-}"
OUTDIR="${2:-$HERE/heatmaps}"

# Avoid the ~/.matplotlib LaTeX-cache PermissionError on this machine.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_bro}"
mkdir -p "$MPLCONFIGDIR" "$OUTDIR"

# Select CSVs: a specific shot count, or all of them.
if [[ -n "$SHOTS" ]]; then
    pattern="$HERE/benchmark_bro_results_p*_s${SHOTS}.csv"
else
    pattern="$HERE/benchmark_bro_results_p*_s*.csv"
fi

# Build a comma-separated, non-empty CSV list (skip header-only files).
csvs=()
for f in $pattern; do
    [[ -f "$f" ]] || continue
    [[ $(wc -l < "$f") -gt 1 ]] || continue   # skip empty (header only)
    csvs+=("$f")
done

if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "No non-empty CSVs matching: $pattern"
    exit 1
fi

joined=$(IFS=,; echo "${csvs[*]}")
echo "Plotting ${#csvs[@]} CSV(s)${SHOTS:+ for shots=$SHOTS} -> $OUTDIR"

./.venv/bin/python "$HERE/plot_bro_heatmap.py" \
    --csv "$joined" \
    --metric p_failed_round \
    --outdir "$OUTDIR"
