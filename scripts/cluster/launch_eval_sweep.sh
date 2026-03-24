#!/bin/bash
# ============================================================================
# SAM3 Evaluation Sweep Launcher — sbatch to H100
#
# Usage:
#   bash scripts/launch_eval_sweep.sh              # submit all
#   bash scripts/launch_eval_sweep.sh --dry        # dry run (print sbatch cmds)
#   bash scripts/launch_eval_sweep.sh --variant v3_noise_noaug_canary  # one variant
#   bash scripts/launch_eval_sweep.sh --epochs 10 50 100              # selected epochs
#
# Env:
#   WALLTIME  — override per-job walltime (default 01:30:00)
#   OUT_ROOT  — override output root (default outputs/eval_sam3_sweep)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SWEEP_ROOT="$PROJECT_ROOT/scratch/sweep_20260311"
SLURM_TEMPLATE="$SCRIPT_DIR/eval_sweep.slurm"
OUT_ROOT="${OUT_ROOT:-outputs/eval_sam3_sweep}"
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
            head -12 "$0" | tail -8
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
echo "Variants: ${VARIANTS[*]}"
echo "Epochs:   ${EPOCHS[*]} + final"
echo "Walltime: $WALLTIME"
echo "Output:   $OUT_ROOT"
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

    SBATCH_ARGS=(
        "--job-name=eval_${eval_name}"
        "--output=logs/sweep/eval_${eval_name}_%j.out"
        "--error=logs/sweep/eval_${eval_name}_%j.err"
        "--time=${WALLTIME}"
        "--export=ALL,EVAL_NAME=${eval_name},CHECKPOINT=${ckpt_file},OUTPUT_DIR=${out_dir}"
    )

    echo ""
    echo "--- ${variant} / ${epoch_label} ---"
    echo "  Checkpoint: ${ckpt_file}"
    echo "  Output:     ${out_dir}"

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
