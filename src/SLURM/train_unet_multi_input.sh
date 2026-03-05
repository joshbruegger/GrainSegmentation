#!/bin/bash
#SBATCH --job-name=GrainSegTrain
#SBATCH --output=logs/unet-train-%j.log
#SBATCH --mem=256G
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=00:10:00

# SHOULD BE 10H, short is for testing that everything is working before running the full job

set -euo pipefail

# Help function
function usage {
    echo "Usage: $0 [-n <num_inputs>] [-s <image_suffixes>] [-r <run_name>] [-o <output_model>]"
    echo "  -n <number>: Number of inputs (default: 7)"
    echo "  -s <string>: Image suffixes separated by spaces (default: '_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6')"
    echo "  -r <string>: Run name (default: '7in_PPL_AllPPX')"
    echo "  -o <path>: Output model path (optional)"
    exit 1
}

NUM_INPUTS=7
IMAGE_SUFFIXES="_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6"
RUN_NAME="7in_PPL_AllPPX"
OUTPUT_MODEL=""

# Process flags
while getopts ":n:s:r:o:h" opt; do
    case $opt in
        n) NUM_INPUTS="$OPTARG";;
        s) IMAGE_SUFFIXES="$OPTARG";;
        r) RUN_NAME="$OPTARG";;
        o) OUTPUT_MODEL="$OPTARG";;
        h|\?) usage;;
    esac
done

source src/SLURM/prepare_env.sh

if [ -z "$OUTPUT_MODEL" ]; then
    OUTPUT_MODEL="$SCRATCH/GrainSeg/models/unet_finetuned_${RUN_NAME}.keras"
fi

echo "Copying dataset to TMPDIR..."
mkdir -p "$TMPDIR/dataset"
cp -r "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped" "$TMPDIR/dataset/"

LOCAL_DIR="$TMPDIR/dataset/cropped"

# Suppress TensorFlow info logs
export TF_CPP_MIN_LOG_LEVEL=2

echo "Running training..."
uv run --no-sync python -u src/train_unet_multi_input.py \
    --run-name "${SLURM_JOB_ID:-local}_${RUN_NAME}" \
    --tuning-dir "$TMPDIR/tuning" \
    --image-dir "$LOCAL_DIR" \
    --mask-dir "$LOCAL_DIR" \
    --checkpoint models/pretrained/starting_point.keras \
    --output-model "$OUTPUT_MODEL" \
    --patch-size 3008 \
    --patch-overlap 0.5 \
    --epochs 100 \
    --tune-epochs 20 \
    --num-inputs "$NUM_INPUTS" \
    --image-suffixes $IMAGE_SUFFIXES \
    --mask-ext .tif \
    --mask-stem-suffix _labels

echo "Copying tuning logs back to SCRATCH..."
mkdir -p "$SCRATCH/GrainSeg/tuning_logs"
cp -r "$TMPDIR/tuning" "$SCRATCH/GrainSeg/tuning_logs/"
echo "Done."
