#!/bin/bash
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

# ---- job-name, output, error, time are set by launch_eval_sweep.sh via sbatch CLI ----
# Required env vars from launcher: EVAL_NAME, CHECKPOINT, OUTPUT_DIR
#
# Runs clean + 4 noisy tiers per checkpoint, then produces comparison charts.
# Output layout:
#   ${OUTPUT_DIR}/clean/eval_results_*.json
#   ${OUTPUT_DIR}/snr05/eval_results_*.json
#   ${OUTPUT_DIR}/snr10/eval_results_*.json
#   ${OUTPUT_DIR}/snr20/eval_results_*.json
#   ${OUTPUT_DIR}/snr50/eval_results_*.json
#   ${OUTPUT_DIR}/comparison/  (bar charts, degradation curves, summary.csv)

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Job Name:    $SLURM_JOB_NAME"
echo "Eval Name:   $EVAL_NAME"
echo "Checkpoint:  $CHECKPOINT"
echo "Output Dir:  $OUTPUT_DIR"
echo "Node:        $SLURMD_NODENAME"
echo "Start Time:  $(date)"
echo "============================================"

mkdir -p logs/sweep

echo "${EVAL_NAME}  job=${SLURM_JOB_ID}  node=${SLURMD_NODENAME}  started=$(date -Iseconds)" >> logs/sweep/eval_manifest.txt

cd /home/yuqyan/Yuqi/LSB-AI-Detection

module load miniforge3

echo "GPU Info:"
nvidia-smi

VARIANT="asinh_stretch"
CONFIG="configs/eval_sam3.yaml"
FAILED=0

# --- 0. Clean ---
echo ""
echo "===== [0/5] Clean ====="
conda run --no-capture-output -n sam3 python scripts/evaluate_sam3.py \
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
    conda run --no-capture-output -n sam3 python scripts/evaluate_sam3.py \
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
conda run --no-capture-output -n sam3 python scripts/visualize_eval_metrics.py \
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
