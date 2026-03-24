#!/bin/bash
# ============================================================
# SAM3 Sweep Checkpoint Evaluation
# ============================================================
# Iterate over all checkpoints in sweep_20260311 variants,
# evaluate on clean data, write to checkpoint-tagged output dirs.
#
# Usage:
#   conda run -n sam3 --no-capture-output bash scripts/eval/run_sweep_eval.sh
#
# Failure strategy: continue-on-error (matches run_batch_eval_type_aware.sh).
# Failed runs are logged to stdout but do not block subsequent checkpoints.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ROOT="scratch/sweep_20260311"
CONFIG="configs/eval_sam3.yaml"
OUT_ROOT="outputs/eval_sam3_sweep"
VARIANTS=("v3_noise_noaug_canary" "v4_noise_aug_canary")
EPOCHS=(10 20 30 40 50 60 70 80 90 100)

TOTAL=0
FAILED=0

for variant in "${VARIANTS[@]}"; do
  CKPT_DIR="$SWEEP_ROOT/$variant/checkpoints"

  # --- Intermediate checkpoints ---
  for epoch in "${EPOCHS[@]}"; do
    CKPT_FILE="$CKPT_DIR/checkpoint_${epoch}.pt"
    OUT_DIR="$OUT_ROOT/${variant}/epoch_${epoch}"

    if [[ ! -f "$CKPT_FILE" ]]; then
      echo "SKIP: $CKPT_FILE not found"
      continue
    fi

    echo "====== $variant / epoch_${epoch} ======"
    TOTAL=$((TOTAL + 1))

    python scripts/eval/evaluate_sam3.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT_FILE" \
        --output-dir "$OUT_DIR" \
        --save-overlays \
        || { echo "FAILED: $variant epoch_${epoch}, continuing..."; FAILED=$((FAILED + 1)); }
  done

  # --- Final checkpoint ---
  CKPT_FILE="$CKPT_DIR/checkpoint.pt"
  OUT_DIR="$OUT_ROOT/${variant}/final"

  if [[ ! -f "$CKPT_FILE" ]]; then
    echo "SKIP: $CKPT_FILE (final) not found"
    continue
  fi

  echo "====== $variant / final ======"
  TOTAL=$((TOTAL + 1))

  python scripts/eval/evaluate_sam3.py \
      --config "$CONFIG" \
      --checkpoint "$CKPT_FILE" \
      --output-dir "$OUT_DIR" \
      --save-overlays \
      || { echo "FAILED: $variant final, continuing..."; FAILED=$((FAILED + 1)); }
done

echo "======================================================"
echo " Sweep complete: ${TOTAL} runs, ${FAILED} failures"
echo " Results in: ${OUT_ROOT}/"
echo "======================================================"
