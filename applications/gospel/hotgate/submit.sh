#!/bin/bash
# SLURM array job — one task per (circuit, noise level) pair.
#
# Submit:
#   sbatch applications/gospel/hotgate/submit.sh
#
# After all tasks finish, plot each noise level:
#   for p in 0.1 0.01 0.02 0.05 0.07 0.005; do
#     python applications/gospel/hotgate/plot.py \
#       --n-qubits 5 --depth 5 --p-ent $p \
#       --results-dir applications/gospel/hotgate/results/p${p}
#   done

#SBATCH --job-name=hotgate
#SBATCH --array=0-599             # 6 noise levels × 100 circuits = 600 tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=applications/gospel/hotgate/logs/slurm_%A_%a.out
#SBATCH --error=applications/gospel/hotgate/logs/slurm_%A_%a.err

# ── environment ───────────────────────────────────────────────────────────────
# Activate your Python environment here, e.g.:
#   source /path/to/venv/bin/activate
# or with conda:
#   conda activate veriphix

# ── parameters ────────────────────────────────────────────────────────────────
N_QUBITS=5
DEPTH=5
BQP_ERROR="1e-1"
N_TEST_ROUNDS=100
BASE_SEED=42
N_CIRCUITS=100

NOISE_VALUES=(0.2 0.3 0.4 0.5)

# ── decompose task ID into (noise_idx, circuit_idx) ───────────────────────────
NOISE_IDX=$(( SLURM_ARRAY_TASK_ID / N_CIRCUITS ))
CIRCUIT_IDX=$(( SLURM_ARRAY_TASK_ID % N_CIRCUITS ))
P_ENT=${NOISE_VALUES[$NOISE_IDX]}
OUT_DIR="applications/gospel/hotgate/results/p${P_ENT}"

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p applications/gospel/hotgate/logs

python applications/gospel/hotgate/simulate.py \
    --circuit-idx   "$CIRCUIT_IDX" \
    --n-qubits      "$N_QUBITS" \
    --depth         "$DEPTH" \
    --bqp-error     "$BQP_ERROR" \
    --n-test-rounds "$N_TEST_ROUNDS" \
    --p-ent         "$P_ENT" \
    --base-seed     "$BASE_SEED" \
    --out-dir       "$OUT_DIR"
