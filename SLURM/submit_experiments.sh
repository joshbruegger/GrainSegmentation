#!/bin/bash

# Help function
function usage {
    echo "Usage: $0 [-p] [-b] [-x] [-a] [-c] [-t]"
    echo "  -p: submit PPL only (1 input) job"
    echo "  -b: submit PPL + PPX Blended (2 inputs) job"
    echo "  -x: submit PPL + All PPX (7 inputs) job"
    echo "  -a: submit ALL jobs"
    echo "  -c: continue/resume from latest model if it exists"
    echo "  -t: skip tuning"
    echo "  combination of flags is possible (e.g. -pxc)"
    exit 1
}

run_ppl=false
run_blended=false
run_all_ppx=false
continue_run=""
skip_tuning=""

# Process flags
while getopts ":pbxact" opt; do
    case $opt in
        p) run_ppl=true;;
        b) run_blended=true;;
        x) run_all_ppx=true;;
        a) run_ppl=true
           run_blended=true
           run_all_ppx=true;;
        c) continue_run="-c";;
        t) skip_tuning="-t";;
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
        --job-name=Train_PPL \
        --cpus-per-task=8 \
        SLURM/train_unet_multi_input.sh -n 1 -s "_PPL" -r "1in_PPL" $continue_run $skip_tuning
    submitted=true
fi

if [ "$run_blended" = true ]; then
    echo "Submitting PPL + PPX Blended (2 inputs) job..."
    sbatch \
        --job-name=Train_PPL_Blended \
        --cpus-per-task=16 \
        SLURM/train_unet_multi_input.sh -n 2 -s "_PPL _PPX_blended" -r "2in_PPL_Blended" $continue_run $skip_tuning
    submitted=true
fi

if [ "$run_all_ppx" = true ]; then
    echo "Submitting PPL + All PPX (7 inputs) job..."
    sbatch \
        --job-name=Train_PPL_AllPPX \
        --cpus-per-task=32 \
        --time=10:00:00 \
        SLURM/train_unet_multi_input.sh -n 7 -s "_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6" -r "7in_PPL_AllPPX" $continue_run $skip_tuning
    submitted=true
fi

if [ "$submitted" = true ]; then
    echo "Selected jobs submitted successfully!"
fi