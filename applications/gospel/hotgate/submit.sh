#!/bin/bash
# SLURM array job — one task per (circuit, probability) pair.
#
# Submit:
#   sbatch applications/gospel/hotgate/submit.sh
#
# After all tasks finish, plot:
#   bash applications/gospel/hotgate/plot_all.sh

#SBATCH --job-name=hotgate
#SBATCH --array=0-499             # N_PROB_VALUES × N_CIRCUITS - 1  (update if you change PROB_VALUES)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=applications/gospel/hotgate/logs/slurm_%A_%a.out
#SBATCH --error=applications/gospel/hotgate/logs/slurm_%A_%a.err

# ── environment ───────────────────────────────────────────────────────────────
# source /path/to/venv/bin/activate

# ── parameters ────────────────────────────────────────────────────────────────
N_QUBITS=5
DEPTH=5
BQP_ERROR="1e-1"
N_TEST_ROUNDS=100
BASE_SEED=42
N_CIRCUITS=100
MALICIOUS_N_NODES=10

# ── noise mode: "depolarising" or "malicious" ─────────────────────────────────
NOISE_MODEL="malicious"

# PROB_VALUES = p_ent for depolarising, malicious_prob for malicious.
# Update --array upper bound to N_PROB_VALUES × N_CIRCUITS - 1 when changing this.
PROB_VALUES=(0.1 0.2 0.3 0.4 0.5)

# ── decompose task ID into (prob_idx, circuit_idx) ────────────────────────────
PROB_IDX=$(( SLURM_ARRAY_TASK_ID / N_CIRCUITS ))
CIRCUIT_IDX=$(( SLURM_ARRAY_TASK_ID % N_CIRCUITS ))
PROB=${PROB_VALUES[$PROB_IDX]}
OUT_DIR="applications/gospel/hotgate/results/${NOISE_MODEL}_p${PROB}"

# ── noise-model-specific args ─────────────────────────────────────────────────
if [[ "$NOISE_MODEL" == "malicious" ]]; then
    NOISE_ARGS="--malicious-n-nodes $MALICIOUS_N_NODES --malicious-prob $PROB"
else
    NOISE_ARGS="--p-ent $PROB"
fi

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p applications/gospel/hotgate/logs

python applications/gospel/hotgate/simulate.py \
    --circuit-idx   "$CIRCUIT_IDX" \
    --n-qubits      "$N_QUBITS" \
    --depth         "$DEPTH" \
    --bqp-error     "$BQP_ERROR" \
    --n-test-rounds "$N_TEST_ROUNDS" \
    --noise-model   "$NOISE_MODEL" \
    --base-seed     "$BASE_SEED" \
    --out-dir       "$OUT_DIR" \
    $NOISE_ARGS
