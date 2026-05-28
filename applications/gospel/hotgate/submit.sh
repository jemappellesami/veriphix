#!/bin/bash
# SLURM array job — one task per circuit.
#
# Submit:
#   sbatch applications/gospel/hotgate/submit.sh
#
# After all tasks finish, plot:
#   python applications/gospel/hotgate/plot.py
#
# Tune the parameters below to match your cluster.

#SBATCH --job-name=hotgate
#SBATCH --array=0-99              # one task per circuit (100 circuits total)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00           # adjust based on observed runtime per circuit
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
P_ENT=0.1
BASE_SEED=42
OUT_DIR="applications/gospel/hotgate/results"

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p applications/gospel/hotgate/logs

python applications/gospel/hotgate/simulate.py \
    --circuit-idx   "$SLURM_ARRAY_TASK_ID" \
    --n-qubits      "$N_QUBITS" \
    --depth         "$DEPTH" \
    --bqp-error     "$BQP_ERROR" \
    --n-test-rounds "$N_TEST_ROUNDS" \
    --p-ent         "$P_ENT" \
    --base-seed     "$BASE_SEED" \
    --out-dir       "$OUT_DIR"
