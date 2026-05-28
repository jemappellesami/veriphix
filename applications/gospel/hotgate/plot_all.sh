#!/bin/bash
# plot_all.sh — produce heatmaps for every results/ subdirectory.
# Outputs are organised as plots/{noise_model}/heatmap_p{prob}.pdf.
#
# Usage:
#   bash applications/gospel/hotgate/plot_all.sh           # skip existing plots
#   bash applications/gospel/hotgate/plot_all.sh --force   # regenerate all

N_QUBITS=5
DEPTH=5
BQP_ERROR="1e-1"
N_TEST_ROUNDS=100
RESULTS_BASE="applications/gospel/hotgate/results"
PLOTS_BASE="applications/gospel/hotgate/plots"

for results_dir in "${RESULTS_BASE}"/*/; do
    dirname=$(basename "$results_dir")

    # Parse noise model and prob from directory name.
    # Supported formats: malicious_p0.3 | depolarising_p0.1 | p0.1 (legacy depolarising)
    if [[ "$dirname" =~ ^(malicious|depolarising)_p(.+)$ ]]; then
        NOISE_MODEL="${BASH_REMATCH[1]}"
        PROB="${BASH_REMATCH[2]}"
    elif [[ "$dirname" =~ ^p(.+)$ ]]; then
        NOISE_MODEL="depolarising"
        PROB="${BASH_REMATCH[1]}"
    else
        echo "[SKIP] $dirname — unrecognised directory name format"
        continue
    fi

    pdf="${PLOTS_BASE}/${NOISE_MODEL}/heatmap_p${PROB}.pdf"

    if [[ -f "$pdf" ]] && [[ "${1}" != "--force" ]]; then
        echo "[SKIP] $dirname — $pdf already exists (use --force to regenerate)"
        continue
    fi

    count=$(find "$results_dir" -name "circuit_*.csv" 2>/dev/null | wc -l)
    if [[ "$count" -eq 0 ]]; then
        echo "[SKIP] $dirname — no CSVs found"
        continue
    fi

    mkdir -p "${PLOTS_BASE}/${NOISE_MODEL}"
    echo "[PLOT] $dirname ($count CSVs) → $pdf"

    if [[ "$NOISE_MODEL" == "malicious" ]]; then
        NOISE_ARGS="--malicious-prob $PROB"
    else
        NOISE_ARGS="--p-ent $PROB"
    fi

    python applications/gospel/hotgate/plot.py \
        --n-qubits      "$N_QUBITS" \
        --depth         "$DEPTH" \
        --bqp-error     "$BQP_ERROR" \
        --n-test-rounds "$N_TEST_ROUNDS" \
        --noise-model   "$NOISE_MODEL" \
        --results-dir   "$results_dir" \
        --out           "$pdf" \
        $NOISE_ARGS
done
