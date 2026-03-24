#!/bin/bash
# ============================================================================
# SAM3 Evaluation Sweep Launcher — sbatch to H100
#
# Usage (from anywhere):
#   bash scripts/cluster/launch_eval_sweep.sh              # submit all
#   bash scripts/cluster/launch_eval_sweep.sh --dry        # dry run (print sbatch cmds)
#   bash scripts/cluster/launch_eval_sweep.sh --variant v3_noise_noaug_canary
#   bash scripts/cluster/launch_eval_sweep.sh --epochs 10 50 100
#
# Env:
#   REPO_ROOT — override repo path (default: auto from this script’s location)
#   SWEEP_ROOT — checkpoint root (default: $REPO_ROOT/scratch/sweep_20260311)
#   WALLTIME  — per-job walltime (default 06:00:00)
#   OUT_ROOT  — eval output root under repo (default outputs/eval_sam3_sweep)
#   CONDA_ENV — conda env name passed to the Slurm job (default sam3)
#
# Cluster portability:
#   Edit scripts/cluster/eval_sweep.slurm (#SBATCH, module load, mail) for your site.
#   The launcher only needs sbatch + paths to data/checkpoints under REPO_ROOT.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/scratch/sweep_20260311}"
SLURM_TEMPLATE="$SCRIPT_DIR/eval_sweep.slurm"
OUT_ROOT="${OUT_ROOT:-outputs/eval_sam3_sweep}"
CONDA_ENV="${CONDA_ENV:-sam3}"

cd "$REPO_ROOT"
WALLTIME="${WALLTIME:-06:00:00}"

ALL_VARIANTS=("v3_noise_noaug_canary" "v4_noise_aug_canary")
ALL_EPOCHS=(10 20 30 40 50 60 70 80 90 100)

DRY_RUN=false
SELECTED_VARIANTS=()
SELECTED_EPOCHS=()

# --- Parse args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --dry)
            DRY_RUN=true
            shift
            ;;
        --variant)
            shift
            while [ $# -gt 0 ] && [[ ! "$1" == --* ]]; do
                SELECTED_VARIANTS+=("$1")
                shift
            done
            ;;
        --epochs)
            shift
            while [ $# -gt 0 ] && [[ ! "$1" == --* ]]; do
                SELECTED_EPOCHS+=("$1")
                shift
            done
            ;;
        --help|-h)
            head -22 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

VARIANTS=("${SELECTED_VARIANTS[@]:-${ALL_VARIANTS[@]}}")
EPOCHS=("${SELECTED_EPOCHS[@]:-${ALL_EPOCHS[@]}}")

mkdir -p logs/sweep
SUBMITTED=0
SKIPPED=0

echo "============================================"
echo "SAM3 Eval Sweep Launcher"
echo "Repo:     $REPO_ROOT"
echo "Variants: ${VARIANTS[*]}"
echo "Epochs:   ${EPOCHS[*]} + final"
echo "Walltime: $WALLTIME"
echo "Output:   $OUT_ROOT"
echo "Conda:    $CONDA_ENV"
echo "============================================"

submit_job() {
    local variant="$1"
    local epoch_label="$2"
    local ckpt_file="$3"

    if [[ ! -f "$ckpt_file" ]]; then
        echo "  SKIP: $ckpt_file not found"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local eval_name="${variant}__${epoch_label}"
    local out_dir="${OUT_ROOT}/${variant}/${epoch_label}"

    # REPO_ROOT + CONDA_ENV so the batch script does not depend on hardcoded paths.
    SBATCH_ARGS=(
        "--job-name=eval_${eval_name}"
        "--output=logs/sweep/eval_${eval_name}_%j.out"
        "--error=logs/sweep/eval_${eval_name}_%j.err"
        "--time=${WALLTIME}"
        "--export=ALL,EVAL_NAME=${eval_name},CHECKPOINT=${ckpt_file},OUTPUT_DIR=${out_dir},REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV}"
    )

    echo ""
    echo "--- ${variant} / ${epoch_label} ---"
    echo "  Checkpoint: $ckpt_file"
    echo "  Output:     $out_dir"

    if [ "$DRY_RUN" = true ]; then
        printf '  [DRY RUN] sbatch'
        printf ' %q' "${SBATCH_ARGS[@]}"
        printf ' %q\n' "${SLURM_TEMPLATE}"
    else
        JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SLURM_TEMPLATE}" | awk '{print $NF}')"
        echo "  Submitted: SLURM Job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    fi
}

for variant in "${VARIANTS[@]}"; do
    CKPT_DIR="$SWEEP_ROOT/$variant/checkpoints"

    # Intermediate checkpoints
    for epoch in "${EPOCHS[@]}"; do
        submit_job "$variant" "epoch_${epoch}" "$CKPT_DIR/checkpoint_${epoch}.pt"
    done

    # Final checkpoint
    submit_job "$variant" "final" "$CKPT_DIR/checkpoint.pt"
done

echo ""
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. No jobs submitted. (${SKIPPED} skipped)"
else
    echo "Submitted ${SUBMITTED} job(s). (${SKIPPED} skipped)"
    echo "Monitor: squeue -u \$USER"
    echo "Manifest: cat logs/sweep/eval_manifest.txt"
fi
echo "Results will be in: ${OUT_ROOT}/"
echo "============================================"
