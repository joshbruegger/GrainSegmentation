#!/bin/bash
#SBATCH --job-name=GrainSegTrain
#SBATCH --output=logs/unet-train-%j.log
#SBATCH --mem=120G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$PROJECT_ROOT/logs"

source prepare_env.sh

# Paths can be overridden via environment variables
IMAGE_DIR="${IMAGE_DIR:-$SCRATCH/GrainSeg/train/images}"
MASK_DIR="${MASK_DIR:-$SCRATCH/GrainSeg/train/masks}"
CHECKPOINT="${CHECKPOINT:-}"
RESUME="${RESUME:-}"
OUTPUT_MODEL="${OUTPUT_MODEL:-$SCRATCH/GrainSeg/models/unet_7in_finetuned.keras}"

PATCH_SIZE="${PATCH_SIZE:-3008}"
STRIDE="${STRIDE:-1504}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-100}"
IMG1_SUFFIX="${IMG1_SUFFIX:-_img1}"
IMG_SUFFIX_TEMPLATE="${IMG_SUFFIX_TEMPLATE:-_img{index}}"
MASK_EXT="${MASK_EXT:-}"
MASK_STEM_SUFFIX="${MASK_STEM_SUFFIX:-}"
SPATIAL_SPLIT="${SPATIAL_SPLIT:-}"
SPLIT_TILE_SIZE="${SPLIT_TILE_SIZE:-}"
SPLIT_COVERAGE_BINS="${SPLIT_COVERAGE_BINS:-}"

echo "Copying training data to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/train_unet_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
cp -r "$IMAGE_DIR" "$WORK_DIR/images"
cp -r "$MASK_DIR" "$WORK_DIR/masks"

LOCAL_IMAGE_DIR="$WORK_DIR/images"
LOCAL_MASK_DIR="$WORK_DIR/masks"

CMD=(uv run python -u src/train_unet_multi_input.py
    --image-dir "$LOCAL_IMAGE_DIR"
    --mask-dir "$LOCAL_MASK_DIR"
    --output-model "$OUTPUT_MODEL"
    --patch-size "$PATCH_SIZE"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --img1-suffix "$IMG1_SUFFIX"
    --img-suffix-template "$IMG_SUFFIX_TEMPLATE"
)
if [ -n "$MASK_EXT" ]; then
    CMD+=(--mask-ext "$MASK_EXT")
fi
if [ -n "$MASK_STEM_SUFFIX" ]; then
    CMD+=(--mask-stem-suffix "$MASK_STEM_SUFFIX")
fi
if [ -n "$SPATIAL_SPLIT" ]; then
    CMD+=(--spatial-split)
fi
if [ -n "$SPLIT_TILE_SIZE" ]; then
    CMD+=(--split-tile-size "$SPLIT_TILE_SIZE")
fi
if [ -n "$SPLIT_COVERAGE_BINS" ]; then
    CMD+=(--split-coverage-bins "$SPLIT_COVERAGE_BINS")
fi

if [ -n "$CHECKPOINT" ]; then
    CMD+=(--checkpoint "$CHECKPOINT")
elif [ -n "$RESUME" ]; then
    CMD+=(--resume "$RESUME")
fi

echo "Running training..."
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
