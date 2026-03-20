#!/bin/bash
# Submit one TuneWatershed job per U-Net input configuration (parallel).
# Presets match infer_model_config in SLURM/evaluate_models_and_plot.sh.
# Align model basenames with train_unet_multi_input.sh --run-name outputs under $GRAINSEG_ROOT/models/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAINSEG_ROOT="${SCRATCH:-/scratch/${USER}}/GrainSeg"
DRY_RUN=false

function usage {
    echo "Usage: $0 [--dry-run] [--help]"
    echo "  Submits four sbatch jobs (PPL, PPLPPXblend, PPL+PPXblend, PPL+AllPPX)."
    echo "  Edit model basenames in this script if your unet_finetuned_*.keras names differ."
    echo "  --dry-run   Print sbatch commands without submitting"
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

submit_one() {
    local job_name="$1"
    local model_basename="$2"
    local num_inputs="$3"
    local suffixes="$4"
    local out_subdir="$5"

    local model_path="$GRAINSEG_ROOT/models/$model_basename"
    local out_dir="$GRAINSEG_ROOT/runs/watershed_tune/$out_subdir"

    local -a cmd=(
        sbatch
        "--job-name=$job_name"
        "$REPO_ROOT/SLURM/tune_watershed.sh"
        --model-path "$model_path"
        --num-inputs "$num_inputs"
        --image-suffixes "$suffixes"
        --output-dir "$out_dir"
    )

    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}

# Model basenames must match unet_finetuned_${RUN_NAME}.keras from training.
submit_one "TuneWatershed_PPL" "unet_finetuned_PPL.keras" 1 "_PPL" "PPL"
submit_one "TuneWatershed_PPLPPXblend" "unet_finetuned_PPLPPXblend.keras" 1 "_PPLPPXblend" "PPLPPXblend"
submit_one "TuneWatershed_PPL_PPXblend" "unet_finetuned_PPL+PPXblend.keras" 2 "_PPL _PPXblend" "PPL_PlusPPXblend"
submit_one "TuneWatershed_PPL_AllPPX" "unet_finetuned_PPL+AllPPX.keras" 7 "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" "PPL_AllPPX"

if [ "$DRY_RUN" = false ]; then
    echo "Submitted watershed tuning jobs for all U-Net variants."
fi
