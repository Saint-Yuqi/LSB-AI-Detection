#!/bin/bash
set -eo pipefail

echo "=================================================="
echo " SAM3 Batch Evaluation (Clean + Noisy Tiers) [Type-Aware]"
echo "=================================================="

VARIANT="asinh_stretch"

echo "0. Run Clean Baseline"
python scripts/evaluate_sam3.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/current/$VARIANT \
    --output-dir outputs/eval_sam3 \
    --snr-tag clean --save-overlays || echo "Clean eval failed but continuing..."

echo "1. Run SNR 05"
python scripts/evaluate_sam3.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr05 \
    --output-dir outputs/eval_sam3_snr05 \
    --snr-tag snr05 --save-overlays || echo "SNR 05 eval failed but continuing..."

echo "2. Run SNR 10"
python scripts/evaluate_sam3.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr10 \
    --output-dir outputs/eval_sam3_snr10 \
    --snr-tag snr10 --save-overlays || echo "SNR 10 eval failed but continuing..."

echo "3. Run SNR 20"
python scripts/evaluate_sam3.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr20 \
    --output-dir outputs/eval_sam3_snr20 \
    --snr-tag snr20 --save-overlays || echo "SNR 20 eval failed but continuing..."

echo "4. Run SNR 50"
python scripts/evaluate_sam3.py \
    --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/noisy/$VARIANT/snr50 \
    --output-dir outputs/eval_sam3_snr50 \
    --snr-tag snr50 --save-overlays || echo "SNR 50 eval failed but continuing..."

echo "=================================================="
echo " All evaluations complete."
echo "=================================================="

echo "5. Run Visualization"
python scripts/visualize_eval_metrics.py \
    --results-dirs outputs/eval_sam3 outputs/eval_sam3_snr05 outputs/eval_sam3_snr10 outputs/eval_sam3_snr20 outputs/eval_sam3_snr50 \
    --labels clean snr05 snr10 snr20 snr50 \
    --output-dir outputs/eval_sam3_comparison
