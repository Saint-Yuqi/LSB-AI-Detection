#!/bin/bash
# =============================================================================
# Batch pNbody physics noise injection: SLURM array (one galaxy per task)
#
# Each task injects noise at all SB_limit levels for all 24 LOS of one galaxy
# (24 views x 2 profiles = 48 CLI calls per task).
#
# Usage:
#   cd /home/yuqyan/Yuqi/LSB-AI-Detection && sbatch scripts/cluster/generate_pnbody_noisy_fits.slurm
# =============================================================================
#SBATCH --job-name=gen_pnbody_noisy
#SBATCH --array=0-183
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=lowprio
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_pnbody_noisy_%A_%a.out
#SBATCH --error=logs/gen_pnbody_noisy_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuqi.yang@uzh.ch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/yuqyan/Yuqi/LSB-AI-Detection}"
cd "$REPO_ROOT"
mkdir -p logs

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:      $(date -Iseconds)"
echo "============================================"

module load miniforge3

GALAXY_IDS=($(conda run -n sam3 python -c "
import yaml
cfg = yaml.safe_load(open('configs/noise_profiles_pnbody_physics.yaml'))
print(' '.join(str(g) for g in cfg['data_selection']['galaxy_ids']))
"))
GID=${GALAXY_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Galaxy ID: $GID"

conda run --no-capture-output -n sam3 python scripts/data/generate_pnbody_noisy_fits.py \
    --config configs/noise_profiles_pnbody_physics.yaml --galaxies "$GID"

echo "============================================"
echo "Done: galaxy=$GID  end=$(date -Iseconds)"
echo "============================================"
