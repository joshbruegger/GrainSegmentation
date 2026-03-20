#!/bin/bash
#SBATCH --job-name=TuneWatershed
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$REPO_ROOT/logs"

# -----------------------------------------------------------------------------
# Hardcoded paths — edit GRAINSEG_ROOT or individual paths for your dataset/model.
# Uses the same cropped thin-section layout as SLURM/train_unet_multi_input.sh.
# -----------------------------------------------------------------------------
GRAINSEG_ROOT="${SCRATCH:-/scratch/${USER}}/GrainSeg"
DATASET_CROPPED="$GRAINSEG_ROOT/dataset/MWD-1#121/cropped"
MODEL_PATH="$GRAINSEG_ROOT/models/unet_finetuned_PPL+AllPPX.keras"
OUTPUT_DIR="$GRAINSEG_ROOT/runs/watershed_tune"

# If non-empty, skip the model and pass --preds-dir (directory of {sample_id}_pred.png).
PREDS_DIR=""

# Inference / mask pairing (match evaluate_models_and_plot.sh defaults)
NUM_INPUTS=7
PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1
MASK_EXT=".tif"
MASK_STEM_SUFFIX="_labels"
IMAGE_SUFFIXES=(_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6)
IMAGE_SUFFIXES_CLI=""

# Default watershed grid (override by editing the array expansions below)
MIN_DISTANCE=(1 3 5)
BOUNDARY_DILATE_ITER=(0 1)
WATERSHED_CONNECTIVITY=(1 2)
MIN_AREA_PX=(0)
EXCLUDE_BORDER=(0 1)

TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"

function usage {
    echo "Usage: $0 [--model-path <path>] [--num-inputs <n>] [--image-suffixes <string>]"
    echo "         [--dataset-cropped <path>] [--output-dir <path>] [--help]"
    echo "  --model-path <path>       U-Net .keras (ignored if PREDS_DIR is set in script)"
    echo "  --num-inputs <n>          Number of input channels (default: 7)"
    echo "  --image-suffixes <str>    Space-separated suffixes, e.g. '_PPL _PPX1 ...'"
    echo "  --dataset-cropped <path>  Cropped dataset directory"
    echo "  --output-dir <path>       Directory for CSV/JSON outputs"
    echo "  PREDS_DIR: edit script to use cached preds instead of the model"
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-inputs)
            NUM_INPUTS="$2"
            shift 2
            ;;
        --image-suffixes)
            IMAGE_SUFFIXES_CLI="$2"
            shift 2
            ;;
        --dataset-cropped)
            DATASET_CROPPED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
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

if [ -n "$IMAGE_SUFFIXES_CLI" ]; then
    read -r -a IMAGE_SUFFIXES <<< "$IMAGE_SUFFIXES_CLI"
fi

function require_file {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

function require_dir {
    local path="$1"
    local message="$2"
    if [ ! -d "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

if [ -z "$PREDS_DIR" ]; then
    require_file "$MODEL_PATH" "Model not found"
else
    require_dir "$PREDS_DIR" "PREDS_DIR is not a directory"
fi

require_dir "$DATASET_CROPPED" "Dataset (cropped) not found"

source "$REPO_ROOT/SLURM/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_DIR="${TMPDIR:-/tmp}/tune_watershed_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_DIR/dataset"
echo "Staging dataset to $WORK_DIR ..."
cp -r "$DATASET_CROPPED" "$WORK_DIR/dataset/"
LOCAL_IMAGE_DIR="$WORK_DIR/dataset/cropped"
LOCAL_MASK_DIR="$WORK_DIR/dataset/cropped"

mkdir -p "$OUTPUT_DIR"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_CSV="$OUTPUT_DIR/watershed_grid_${JOB_TAG}.csv"
OUT_JSON="$OUTPUT_DIR/watershed_best_${JOB_TAG}.json"

cd "$REPO_ROOT/src/training"
echo "Syncing evaluation/training environment..."
uv sync

WHEEL_PATH="$SCRATCH/GrainSeg/wheels/$TF_WHEEL_NAME"
require_file "$WHEEL_PATH" "TensorFlow wheel not found"
echo "Installing TensorFlow wheel..."
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

TUNE_CMD=(
    uv run --no-sync python -u ../evaluation/tune_watershed.py
    --image-dir "$LOCAL_IMAGE_DIR"
    --mask-dir "$LOCAL_MASK_DIR"
    --output-csv "$OUT_CSV"
    --output-json "$OUT_JSON"
    --num-inputs "$NUM_INPUTS"
    --image-suffixes
    "${IMAGE_SUFFIXES[@]}"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --mask-ext "$MASK_EXT"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --min-distance "${MIN_DISTANCE[@]}"
    --boundary-dilate-iter "${BOUNDARY_DILATE_ITER[@]}"
    --watershed-connectivity "${WATERSHED_CONNECTIVITY[@]}"
    --min-area-px "${MIN_AREA_PX[@]}"
    --exclude-border "${EXCLUDE_BORDER[@]}"
)

if [ -n "$PREDS_DIR" ]; then
    TUNE_CMD+=(--preds-dir "$PREDS_DIR")
else
    LOCAL_MODEL="$WORK_DIR/model.keras"
    cp "$MODEL_PATH" "$LOCAL_MODEL"
    TUNE_CMD+=(--model-path "$LOCAL_MODEL")
fi

echo "Running watershed tuning..."
echo "  CSV: $OUT_CSV"
echo "  JSON: $OUT_JSON"
"${TUNE_CMD[@]}"

echo "Done."
