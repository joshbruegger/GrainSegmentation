#!/bin/bash

set -euo pipefail

function usage {
    echo "Usage: $0 [--ppl] [--ppl-ppx-composite] [--ppl-plus-ppx-composite] [--all-ppx] [--all] [--resume] [--tune] [--verbose] [--help]"
    echo "  --ppl: submit PPL-only YOLO job"
    echo "  --ppl-ppx-composite: submit PPLPPXblend YOLO job"
    echo "  --ppl-plus-ppx-composite: submit PPL+PPXblend YOLO job"
    echo "  --all-ppx: submit PPL+AllPPX YOLO job"
    echo "  --all: submit all YOLO jobs"
    echo "  --resume: resume selected jobs from their latest saved checkpoint"
    echo "  --tune: run Ultralytics built-in hyperparameter tuning for the selected jobs"
    echo "  --verbose: forward verbose logging to the train wrapper"
    exit 1
}

run_ppl=false
run_ppl_ppx_composite=false
run_ppl_plus_ppx_composite=false
run_all_ppx=false
resume_args=()
tune_args=()
verbose_args=()
submitted_job_ids=()

rollback_submissions() {
    if [ "${#submitted_job_ids[@]}" -eq 0 ]; then
        return
    fi

    echo "Rolling back submitted YOLO jobs: ${submitted_job_ids[*]}" >&2
    scancel "${submitted_job_ids[@]}"
}

submit_job() {
    local mem="$1"
    local job_name="$2"
    local variant="$3"
    local run_name="$4"
    local batch_size="$5"
    local output
    local job_id

    if ! output=$(
        sbatch \
            --mem="$mem" \
            --job-name="$job_name" \
            SLURM/train_yolo26x_seg.sh \
            --variant "$variant" \
            --run-name "$run_name" \
            "${resume_args[@]}" \
            "${tune_args[@]}" \
            "${verbose_args[@]}" \
            --batch "$batch_size"
    ); then
        rollback_submissions
        return 1
    fi

    echo "$output"
    job_id=$(printf "%s\n" "$output" | awk '/Submitted batch job/ {print $4}' | tail -n 1)
    if [ -n "$job_id" ]; then
        submitted_job_ids+=("$job_id")
    fi
}

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
        --tune)
            tune_args=(--tune)
            shift
            ;;
        --verbose)
            verbose_args=(--verbose)
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

if [ "$run_ppl" = false ] && [ "$run_ppl_ppx_composite" = false ] && [ "$run_ppl_plus_ppx_composite" = false ] && [ "$run_all_ppx" = false ]; then
    usage
fi

submitted=false

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    submit_job 1000G Train_YOLO_PPL+AllPPX PPL+AllPPX PPL+AllPPX 32
    submitted=true
fi

if [ "$run_ppl_plus_ppx_composite" = true ]; then
    echo "Submitting PPL + PPXblend (2 inputs) job..."
    submit_job 350G Train_YOLO_PPL+PPXblend PPL+PPXblend PPL+PPXblend 32
    submitted=true
fi

if [ "$run_ppl_ppx_composite" = true ]; then
    echo "Submitting PPLPPXBlend (1 input) job..."
    submit_job 200G Train_YOLO_PPLPPXblend PPLPPXblend PPLPPXblend 32
    submitted=true
fi

if [ "$run_ppl" = true ]; then
    echo "Submitting PPL only (1 input) job..."
    submit_job 200G Train_YOLO_PPL PPL PPL 32
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected YOLO jobs submitted successfully!"
fi
