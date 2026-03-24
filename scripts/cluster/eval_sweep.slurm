#!/bin/bash
# =============================================================================
# SAM3 eval sweep — one checkpoint: clean + SNR tiers + comparison plots
#
# Intended use: submit via scripts/cluster/launch_eval_sweep.sh (exports REPO_ROOT).
# Manual submit:  cd <repo_root> && export REPO_ROOT=$PWD && sbatch scripts/cluster/eval_sweep.slurm
#
# Required env (set by launcher or you): EVAL_NAME, CHECKPOINT, OUTPUT_DIR
# Optional env: REPO_ROOT — absolute path to this git repo (strongly recommended)
#
# Site-specific #SBATCH lines below MUST be edited for your cluster (partition, GPU, mail).
# The conda module line is UZH-oriented; remove or replace if you use another stack.
# =============================================================================
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

# ---- job-name, output, error, time are set by launch_eval_sweep.sh via sbatch CLI ----
#
# Output layout:
#   ${OUTPUT_DIR}/clean/eval_results_*.json
#   ${OUTPUT_DIR}/snr05/ ...
#   ${OUTPUT_DIR}/comparison/  (bar charts, degradation curves, summary.csv)

echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-}"
echo "Job Name:    ${SLURM_JOB_NAME:-}"
echo "Eval Name:   ${EVAL_NAME:-}"
echo "Checkpoint:  ${CHECKPOINT:-}"
echo "Output Dir:  ${OUTPUT_DIR:-}"
echo "Node:        ${SLURMD_NODENAME:-}"
echo "Start Time:  $(date)"
echo "============================================"

if [[ -z "${EVAL_NAME:-}" || -z "${CHECKPOINT:-}" || -z "${OUTPUT_DIR:-}" ]]; then
    echo "ERROR: Missing EVAL_NAME, CHECKPOINT, and/or OUTPUT_DIR (use launch_eval_sweep.sh)." >&2
    exit 1
fi

set -euo pipefail

# Resolve repository root: prefer explicit REPO_ROOT from launcher, else submit directory.
REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/scripts/eval/evaluate_sam3.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
fi
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/scripts/eval/evaluate_sam3.py" ]]; then
    echo "ERROR: Invalid REPO_ROOT; cannot find scripts/eval/evaluate_sam3.py" >&2
    echo "  Fix: run from repo via bash scripts/cluster/launch_eval_sweep.sh" >&2
    echo "  Or: cd <repo> && export REPO_ROOT=\$PWD before sbatch" >&2
    exit 1
fi
cd "$REPO_ROOT"

mkdir -p logs/sweep

echo "${EVAL_NAME}  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME}  started=$(date -Iseconds)" >> logs/sweep/eval_manifest.txt

# --- Conda stack (edit or remove for your cluster) ---
module load miniforge3

echo "GPU Info:"
nvidia-smi

CONDA_ENV="${CONDA_ENV:-sam3}"
VARIANT="${VARIANT:-asinh_stretch}"
CONFIG="${CONFIG:-configs/eval_sam3.yaml}"
FAILED=0

run_eval() {
    conda run --no-capture-output -n "${CONDA_ENV}" python scripts/eval/evaluate_sam3.py "$@"
}

# --- 0. Clean ---
echo ""
echo "===== [0/5] Clean ====="
run_eval \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --render-dir "data/02_processed/renders/current/$VARIANT" \
    --output-dir "${OUTPUT_DIR}/clean" \
    --snr-tag clean --save-overlays \
    || { echo "FAILED: clean eval"; FAILED=$((FAILED + 1)); }

# --- 1-4. Noisy tiers ---
TIER_IDX=1
for SNR in snr05 snr10 snr20 snr50; do
    echo ""
    echo "===== [${TIER_IDX}/5] ${SNR} ====="
    run_eval \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --render-dir "data/02_processed/renders/noisy/$VARIANT/$SNR" \
        --output-dir "${OUTPUT_DIR}/${SNR}" \
        --snr-tag "$SNR" --save-overlays \
        || { echo "FAILED: $SNR eval"; FAILED=$((FAILED + 1)); }
    TIER_IDX=$((TIER_IDX + 1))
done

# --- 5. Comparison visualization ---
echo ""
echo "===== [5/5] Comparison Visualization ====="
conda run --no-capture-output -n "${CONDA_ENV}" python scripts/viz/visualize_eval_metrics.py \
    --results-dirs "${OUTPUT_DIR}/clean" "${OUTPUT_DIR}/snr05" \
                   "${OUTPUT_DIR}/snr10" "${OUTPUT_DIR}/snr20" "${OUTPUT_DIR}/snr50" \
    --labels clean snr05 snr10 snr20 snr50 \
    --output-dir "${OUTPUT_DIR}/comparison" \
    || { echo "FAILED: visualization"; FAILED=$((FAILED + 1)); }

echo ""
echo "============================================"
echo "End Time:    $(date)"
echo "Eval Name:   $EVAL_NAME"
echo "Failures:    $FAILED / 6"
echo "============================================"

# Exit 0 even with partial failures (results already written)
exit 0
