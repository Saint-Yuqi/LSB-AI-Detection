#!/bin/bash
# =============================================================================
# Batch-generate PNbody clean FITS: SLURM array (one galaxy per task)
#
# Each task generates 24 LOS FITS for a single galaxy via
# generate_pnbody_fits.py.  No skip-exists: reruns recompute all outputs.
#
# Usage:
#   cd /home/yuqyan/Yuqi/LSB-AI-Detection && sbatch scripts/cluster/generate_pnbody_fits.slurm
# =============================================================================
#SBATCH --job-name=gen_pnbody_fits
#SBATCH --array=0-183
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_pnbody_fits_%A_%a.out
#SBATCH --error=logs/gen_pnbody_fits_%A_%a.err
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
cfg = yaml.safe_load(open('configs/pnbody/firebox_pnbody_24los.yaml'))
print(' '.join(str(g) for g in cfg['galaxy_ids']))
"))
GID=${GALAXY_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Galaxy ID: $GID"

conda run --no-capture-output -n sam3 python scripts/data/generate_pnbody_fits.py \
    --config configs/pnbody/firebox_pnbody_24los.yaml --galaxies "$GID"

echo "============================================"
echo "Done: galaxy=$GID  end=$(date -Iseconds)"
echo "============================================"
