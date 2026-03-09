#!/bin/bash

# Help function
function usage {
    echo "Usage: $0 [--ppl] [--ppl-ppx-composite] [--ppl-plus-ppx-composite] [--all-ppx] [--all] [--resume] [--skip-tuning] [--help]"
    echo "  --ppl: submit PPL-only (1 input) job"
    echo "  --ppl-ppx-composite: submit single PPL+PPX composite (1 input) job"
    echo "  --ppl-plus-ppx-composite: submit PPL + PPX composite (2 inputs) job"
    echo "  --all-ppx: submit PPL + all PPX images (7 inputs) job"
    echo "  --all: submit all jobs"
    echo "  --resume: resume selected jobs from their latest saved model if it exists"
    echo "  --skip-tuning: skip tuning for selected jobs"
    echo "  Combination of flags is allowed."
    exit 1
}

run_ppl=false
run_ppl_ppx_composite=false
run_ppl_plus_ppx_composite=false
run_all_ppx=false
resume_args=()
skip_tuning_args=()

# Process flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ppl)
            run_ppl=true
            shift
            ;;
        --ppl-ppx-composite)
            run_ppl_ppx_composite=true
            shift
            ;;
        --ppl-plus-ppx-composite)
            run_ppl_plus_ppx_composite=true
            shift
            ;;
        --all-ppx)
            run_all_ppx=true
            shift
            ;;
        --all)
            run_ppl=true
            run_ppl_ppx_composite=true
            run_ppl_plus_ppx_composite=true
            run_all_ppx=true
            shift
            ;;
        --resume)
            resume_args=(--resume)
            shift
            ;;
        --skip-tuning)
            skip_tuning_args=(--skip-tuning)
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# If no flags provided, show usage
if [ "$run_ppl" = false ] && [ "$run_ppl_ppx_composite" = false ] && [ "$run_ppl_plus_ppx_composite" = false ] && [ "$run_all_ppx" = false ]; then
    usage
fi

submitted=false

if [ "$run_ppl" = true ]; then
    echo "Submitting PPL only (1 input) job..."
    sbatch \
        --job-name=Train_PPL \
        SLURM/train_unet_multi_input.sh \
        --num-inputs 1 \
        --image-suffixes "_PPL" \
        --run-name "PPL" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}"
    submitted=true
fi

if [ "$run_ppl_ppx_composite" = true ]; then
    echo "Submitting PPLPPXBlend (1 input) job..."
    sbatch \
        --job-name=Train_PPLPPXBlend \
        SLURM/train_unet_multi_input.sh \
        --num-inputs 1 \
        --image-suffixes "_PPLPPXblend" \
        --run-name "PPLPPXblend" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}"
    submitted=true
fi

if [ "$run_ppl_plus_ppx_composite" = true ]; then
    echo "Submitting PPL + PPXblend (2 inputs) job..."
    sbatch \
        --job-name=Train_PPL+PPXblend \
        SLURM/train_unet_multi_input.sh \
        --num-inputs 2 \
        --image-suffixes "_PPL _PPXblend" \
        --run-name "PPL+PPXblend" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}"
    submitted=true
fi

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    sbatch \
        --job-name=Train_PPL+AllPPX \
        --mem=950G \
        --time=24:00:00 \
        SLURM/train_unet_multi_input.sh \
        --num-inputs 7 \
        --image-suffixes "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" \
        --run-name "PPL+AllPPX" \
        "${resume_args[@]}" \
        "${skip_tuning_args[@]}"
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected jobs submitted successfully!"
fi