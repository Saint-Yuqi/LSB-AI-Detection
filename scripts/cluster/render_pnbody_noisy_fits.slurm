#!/bin/bash
# =============================================================================
# Render all pNbody physics-noisy FITS to 8-bit PNG (linear_magnitude).
#
# Reads:  data/04_noise/pnbody_physics/{sb30,sb31.5}/*.fits.gz
# Writes: data/02_processed/renders/noisy/linear_magnitude/{sb30,sb31.5}/...
#
# Skip-exists is built into render_noisy_fits.py for incremental reruns.
#
# Usage:
#   cd /home/yuqyan/Yuqi/LSB-AI-Detection && sbatch scripts/cluster/render_pnbody_noisy_fits.slurm
# =============================================================================
#SBATCH --job-name=render_pnbody_noisy
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=lowprio
#SBATCH --time=03:00:00
#SBATCH --output=logs/render_pnbody_noisy_%j.out
#SBATCH --error=logs/render_pnbody_noisy_%j.err
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
echo "Noise cfg:  configs/noise_profiles_pnbody_physics.yaml"
echo "Data cfg:   configs/unified_data_prep_pnbody.yaml"
echo "Variant:    linear_magnitude"
echo "============================================"

module load miniforge3

conda run --no-capture-output -n sam3 python scripts/data/render_noisy_fits.py \
    --noise-config configs/noise_profiles_pnbody_physics.yaml \
    --data-config configs/unified_data_prep_pnbody.yaml \
    --variants linear_magnitude

echo "============================================"
echo "End:        $(date -Iseconds)"
echo "============================================"
