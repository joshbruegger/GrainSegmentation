#!/bin/bash
#SBATCH --job-name=GrainSegmentation
#SBATCH --output=logs/sam2-%j.log
#SBATCH --mem=60G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=01:00:00

echo "Loading modules..."
module purge
module load cuDNN/9.10.1.4-CUDA-12.8.0
module load SciPy-bundle/2025.06-gfbf-2025a
module list

source SLURM/prepare_env.sh


INPUT_PATH="${1:-$SCRATCH/GrainSeg/dataset/MWD-1#121_s0c0.tif}"
INPUT_NAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_NAME%.*}"

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/starting_masks_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR/out"

cp "$INPUT_PATH" "$WORK_DIR/"

echo "Running starting masks script on local storage..."
cd src/data_prep && uv run python -u starting_masks.py \
    --input "$WORK_DIR/$INPUT_NAME" \
    --output "$WORK_DIR/out" \
    --tile-size 6144 \
    --visualize-probability 0 \
    --nms-thresh 0.3 \
    --max-mask-coverage 0.5 \
    --load-mask-cache \
    --mask-cache-dir "$SCRATCH/GrainSeg/out"
    # --save-mask-cache


echo "Running polygon conversion on local storage..."
uv run python -u convert_rle_polygon.py rle2json \
    -i "$WORK_DIR/out/${INPUT_STEM}.json" \
    -o "$WORK_DIR/out/${INPUT_STEM}.geojson"

echo "Copying results back to persistent storage..."
mkdir -p "$SCRATCH/GrainSeg/out"
cp "$WORK_DIR/out/${INPUT_STEM}.json" "$SCRATCH/GrainSeg/out/"
cp "$WORK_DIR/out/${INPUT_STEM}.geojson" "$SCRATCH/GrainSeg/out/"

echo "Done!"