#!/bin/bash
# =============================================================================
# Regenerate GT canonical Phase 3: SAM3 inference (satellites + streams eval)
#
# Runs SAM3 text-prompt grounding on all 70 images against the new
# SBlim31.5 streams_instance_map.npy from Phase 2.
#
# Usage:
#   cd /home/yuqyan/Yuqi/LSB-AI-Detection && sbatch scripts/cluster/regen_gt_sam3.slurm
# =============================================================================
#SBATCH --job-name=regen_gt_sam3
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --time=04:00:00
#SBATCH --output=logs/regen_gt_sam3_%j.out
#SBATCH --error=logs/regen_gt_sam3_%j.err
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
echo "Config:     configs/unified_data_prep.yaml"
echo "Phase:      inference (SAM3 evaluate)"
echo "SB thresh:  31.5"
echo "============================================"

module load miniforge3
nvidia-smi

conda run --no-capture-output -n sam3 python scripts/data/prepare_unified_dataset.py \
    --config configs/unified_data_prep.yaml \
    --phase inference \
    --force

echo "============================================"
echo "End:   $(date -Iseconds)"
echo "============================================"
