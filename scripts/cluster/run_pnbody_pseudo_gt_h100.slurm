#!/bin/bash
# =============================================================================
# PNbody pseudo GT generation on H100:
#   1. SAM3 inference for clean + noisy diagnostics
#   2. export authoritative clean pseudo GT
#   3. rebuild dataset with clean-truth / noisy-clone augmentation
#
# Usage:
#   cd /home/yuqyan/Yuqi/LSB-AI-Detection
#   sbatch scripts/cluster/run_pnbody_pseudo_gt_h100.slurm
# =============================================================================
#SBATCH --job-name=pnbody_pseudo_gt
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --output=logs/pnbody_pseudo_gt_%j.out
#SBATCH --error=logs/pnbody_pseudo_gt_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/yuqyan/Yuqi/LSB-AI-Detection}"
cd "$REPO_ROOT"
mkdir -p logs

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:      $(date -Iseconds)"
echo "Config:     configs/unified_data_prep_pnbody.yaml"
echo "Dataset:    pnbody"
echo "Conditions: clean sb30 sb31.5"
echo "Phases:     inference -> export -> rebuild"
echo "============================================"

module load miniforge3
nvidia-smi

conda run --no-capture-output -n sam3 python scripts/data/prepare_unified_dataset.py \
    --config configs/unified_data_prep_pnbody.yaml \
    --dataset pnbody \
    --conditions clean sb30 sb31.5 \
    --phase inference \
    --force

conda run --no-capture-output -n sam3 python scripts/data/prepare_unified_dataset.py \
    --config configs/unified_data_prep_pnbody.yaml \
    --dataset pnbody \
    --conditions clean \
    --phase export \
    --force

conda run --no-capture-output -n sam3 python scripts/data/rebuild_sam3_dataset.py \
    --config configs/unified_data_prep_pnbody.yaml \
    --previous-root data/02_processed/sam3_prepared_previous/pnbody \
    --noise-tags sb30 sb31.5 \
    --noisy-variants linear_magnitude

echo "============================================"
echo "End:        $(date -Iseconds)"
echo "============================================"
