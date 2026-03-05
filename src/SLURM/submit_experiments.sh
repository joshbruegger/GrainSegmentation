#!/bin/bash

# Help function
function usage {
    echo "Usage: $0 [-p] [-b] [-x] [-a]"
    echo "  -p: submit PPL only (1 input) job"
    echo "  -b: submit PPL + PPX Blended (2 inputs) job"
    echo "  -x: submit PPL + All PPX (7 inputs) job"
    echo "  -a: submit ALL jobs"
    echo "  combination of flags is possible (e.g. -px)"
    exit 1
}

run_ppl=false
run_blended=false
run_all_ppx=false

# Process flags
while getopts ":pbxa" opt; do
    case $opt in
        p) run_ppl=true;;
        b) run_blended=true;;
        x) run_all_ppx=true;;
        a) run_ppl=true
           run_blended=true
           run_all_ppx=true;;
        \?) echo "Invalid option -$OPTARG" >&2
            usage;;
    esac
done

# If no flags provided, show usage
if [ "$OPTIND" -eq 1 ]; then
    usage
fi

submitted=false

if [ "$run_ppl" = true ]; then
    echo "Submitting PPL only (1 input) job..."
    sbatch \
        --job-name=GrainSegTrain_PPL \
        src/SLURM/train_unet_multi_input.sh -n 1 -s "_PPL" -r "1in_PPL"
    submitted=true
fi

if [ "$run_blended" = true ]; then
    echo "Submitting PPL + PPX Blended (2 inputs) job..."
    sbatch \
        --job-name=GrainSegTrain_PPL_Blended \
        src/SLURM/train_unet_multi_input.sh -n 2 -s "_PPL _PPX_blended" -r "2in_PPL_Blended"
    submitted=true
fi

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    sbatch \
        --job-name=GrainSegTrain_PPL_AllPPX \
        src/SLURM/train_unet_multi_input.sh -n 7 -s "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" -r "7in_PPL_AllPPX"
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected jobs submitted successfully!"
fi