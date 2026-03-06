#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=12:00:00

set -euo pipefail

# Only attempt to cancel if running as a SLURM job
if [ -n "${SLURM_JOB_NAME:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    OLD_JOBS=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -h -o %i | grep -v "^$SLURM_JOB_ID$" || true)
    
    if [ -n "$OLD_JOBS" ]; then
        echo "Canceling previous jobs with name $SLURM_JOB_NAME: $OLD_JOBS"
        # Word-splitting automatically converts newlines to arguments
        scancel $OLD_JOBS
        
        # Give the old job a few seconds to release file locks (logs, weights, etc.)
        sleep 10
    fi
fi

# Help function
function usage {
    echo "Usage: $0 [-n <num_inputs>] [-s <image_suffixes>] [-r <run_name>] [-o <output_model>] [-c] [-t]"
    echo "  -n <number>: Number of inputs (default: 7)"
    echo "  -s <string>: Image suffixes separated by spaces (default: '_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6')"
    echo "  -r <string>: Run name (default: '7in_PPL_AllPPX')"
    echo "  -o <path>: Output model path (optional)"
    echo "  -c: Continue/resume from latest model if it exists"
    echo "  -t: Skip tuning"
    exit 1
}

NUM_INPUTS=7
IMAGE_SUFFIXES="_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6"
RUN_NAME="7in_PPL_AllPPX"
OUTPUT_MODEL=""
CONTINUE_RUN=0
SKIP_TUNING_FLAG=""
FOLDS=2

# Process flags
while getopts ":n:s:r:o:cht" opt; do
    case $opt in
        n) NUM_INPUTS="$OPTARG";;
        s) IMAGE_SUFFIXES="$OPTARG";;
        r) RUN_NAME="$OPTARG";;
        o) OUTPUT_MODEL="$OPTARG";;
        c) CONTINUE_RUN=1;;
        t) SKIP_TUNING_FLAG="--skip-tuning"; FOLDS=5;;
        h|\?) usage;;
    esac
done

source SLURM/prepare_env.sh

if [ -z "$OUTPUT_MODEL" ]; then
    OUTPUT_MODEL="$SCRATCH/GrainSeg/models/unet_finetuned_${RUN_NAME}.keras"
fi

echo "Copying dataset to TMPDIR..."
mkdir -p "$TMPDIR/dataset"
cp -r "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped" "$TMPDIR/dataset/"

LOCAL_DIR="$TMPDIR/dataset/cropped"

# Suppress TensorFlow info logs
export TF_CPP_MIN_LOG_LEVEL=2

echo "Syncing training environment..."
cd src/training
uv sync

echo "Installing TensorFlow wheel..."
# Install our custom TensorFlow wheel compiled for RTX 6000 Blackwell
# and the necessary exact CUDA dependency versions it needs (per the workaround)
WHEEL_PATH="$SCRATCH/GrainSeg/wheels/tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

LATEST_MODEL="${OUTPUT_MODEL%.keras}_latest.keras"

if [ "$CONTINUE_RUN" -eq 1 ] && [ -f "$LATEST_MODEL" ]; then
    CHECKPOINT_ARGS=("--resume" "$LATEST_MODEL")
    echo "Resuming from latest model: $LATEST_MODEL"
else
    CHECKPOINT_ARGS=("--checkpoint" "../../models/pretrained/starting_point.keras")
fi

echo "Running training..."
uv run --no-sync python -u train_unet_multi_input.py \
    $SKIP_TUNING_FLAG \
    --run-name "${SLURM_JOB_ID:-local}_${RUN_NAME}" \
    --tuning-dir "$SCRATCH/GrainSeg/tuning_logs" \
    --image-dir "$LOCAL_DIR" \
    --mask-dir "$LOCAL_DIR" \
    --folds "$FOLDS" \
    "${CHECKPOINT_ARGS[@]}" \
    --output-model "$OUTPUT_MODEL" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --epochs 100 \
    --tune-epochs 20 \
    --num-inputs "$NUM_INPUTS" \
    --image-suffixes $IMAGE_SUFFIXES \
    --mask-ext .tif \
    --mask-stem-suffix _labels

# echo "Copying tuning logs back to SCRATCH..."
# mkdir -p "$SCRATCH/GrainSeg/tuning_logs"
# cp -r "$TMPDIR/tuning" "$SCRATCH/GrainSeg/tuning_logs/"
echo "Done."
