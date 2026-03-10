#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TF_STDERR_FILTER="$REPO_ROOT/SLURM/filter_tensorflow_stderr.py"

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
    echo "Usage: $0 [--num-inputs <num_inputs>] [--image-suffixes <image_suffixes>] [--run-name <run_name>] [--output-model <output_model>] [--resume [model_path]] [--skip-tuning] [--verbose]"
    echo "  --num-inputs <number>: Number of inputs (default: 7)"
    echo "  --image-suffixes <string>: Image suffixes separated by spaces (default: '_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6')"
    echo "  --run-name <string>: Run name (default: '7in_PPL_AllPPX')"
    echo "  --output-model <path>: Output model path (optional)"
    echo "  --resume [path]: Resume final training from a saved checkpoint (defaults to *_latest.keras)"
    echo "  --skip-tuning: Skip tuning"
    echo "  --verbose: Disable stderr filtering and keep raw TensorFlow/XLA diagnostics"
    echo "  Tuned runs choose final epochs with a frozen-CV pass, capped by the training script's --epochs setting."
    echo "  Re-running with the same run name and tuning dir automatically resumes tuner state."
    exit 1
}

NUM_INPUTS=7
IMAGE_SUFFIXES="_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6"
RUN_NAME="7in_PPL_AllPPX"
OUTPUT_MODEL=""
RESUME_MODEL=""
SKIP_TUNING_FLAG=""
FOLDS=2
VERBOSE=false

# Process flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-inputs)
            NUM_INPUTS="$2"
            shift 2
            ;;
        --image-suffixes)
            IMAGE_SUFFIXES="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --output-model)
            OUTPUT_MODEL="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -gt 1 && "${2:-}" != -* ]]; then
                RESUME_MODEL="$2"
                shift 2
            else
                RESUME_MODEL="__LATEST__"
                shift
            fi
            ;;
        --skip-tuning)
            SKIP_TUNING_FLAG="--skip-tuning"
            FOLDS=5
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

source "$REPO_ROOT/SLURM/prepare_env.sh"

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
cd "$REPO_ROOT/src/training"
uv sync

echo "Installing TensorFlow wheel..."
# Install our custom TensorFlow wheel compiled for RTX 6000 Blackwell
# and the necessary exact CUDA dependency versions it needs (per the workaround)
WHEEL_PATH="$SCRATCH/GrainSeg/wheels/tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

LATEST_MODEL="${OUTPUT_MODEL%.keras}_latest.keras"

if [ "$RESUME_MODEL" = "__LATEST__" ]; then
    RESUME_MODEL="$LATEST_MODEL"
fi

if [ -n "$RESUME_MODEL" ]; then
    if [ ! -f "$RESUME_MODEL" ]; then
        echo "Resume checkpoint not found: $RESUME_MODEL"
        exit 1
    fi
    CHECKPOINT_ARGS=("--resume" "$RESUME_MODEL")
    echo "Resuming final training from: $RESUME_MODEL"
else
    CHECKPOINT_ARGS=("--checkpoint" "../../models/pretrained/starting_point.keras")
fi

echo "Running training..."
read -r -a IMAGE_SUFFIX_ARGS <<< "$IMAGE_SUFFIXES"
TRAIN_CMD=(uv run --no-sync python -u train_unet_multi_input.py)

if [ -n "$SKIP_TUNING_FLAG" ]; then
    TRAIN_CMD+=("$SKIP_TUNING_FLAG")
fi

TRAIN_CMD+=(
    --run-name "$RUN_NAME"
    --tuning-dir "$SCRATCH/GrainSeg/tuning_logs"
    --image-dir "$LOCAL_DIR"
    --mask-dir "$LOCAL_DIR"
    --folds "$FOLDS"
    "${CHECKPOINT_ARGS[@]}"
    --output-model "$OUTPUT_MODEL"
    --patch-size 1024
    --patch-overlap 0.5
    --epochs 100
    --tune-epochs 20
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIX_ARGS[@]}"
    --mask-ext .tif
    --mask-stem-suffix _labels
)

if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled. Raw TensorFlow/XLA stderr will be logged."
    "${TRAIN_CMD[@]}"
else
    "${TRAIN_CMD[@]}" 2> >(python -u "$TF_STDERR_FILTER" >&2)
fi

# echo "Copying tuning logs back to SCRATCH..."
# mkdir -p "$SCRATCH/GrainSeg/tuning_logs"
# cp -r "$TMPDIR/tuning" "$SCRATCH/GrainSeg/tuning_logs/"
echo "Done."
