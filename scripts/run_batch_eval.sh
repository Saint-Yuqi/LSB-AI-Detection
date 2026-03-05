#!/bin/bash
set -eo pipefail

echo "=================================================="
echo " SAM3 Batch Evaluation (Clean + Noisy Tiers)"
echo "=================================================="

# Ensure we're in the right conda env
if [[ "$CONDA_DEFAULT_ENV" != "sam3" ]]; then
    # We use conda run instead since this script will be executed via it
    echo "This script should be run with 'conda run -n sam3 bash ...'"
fi

VARIANT="asinh_stretch"

echo "Wait for clean evaluation to finish (already running in background)..."
# Just waiting for user to run the others

echo "1. Run SNR 05"
python scripts/evaluate_model.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr05 \
    --output-dir outputs/eval_sam3_snr05 \
    --snr-tag snr05 --save-overlays

echo "2. Run SNR 10"
python scripts/evaluate_model.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr10 \
    --output-dir outputs/eval_sam3_snr10 \
    --snr-tag snr10 --save-overlays

echo "3. Run SNR 20"
python scripts/evaluate_model.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr20 \
    --output-dir outputs/eval_sam3_snr20 \
    --snr-tag snr20 --save-overlays

echo "4. Run SNR 50"
python scripts/evaluate_model.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr50 \
    --output-dir outputs/eval_sam3_snr50 \
    --snr-tag snr50 --save-overlays

echo "=================================================="
echo " All evaluations complete."
echo "=================================================="
