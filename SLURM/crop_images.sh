#!/bin/bash
#SBATCH --job-name=crop_images
#SBATCH --output=logs/crop_images-%j.log
#SBATCH --time=00:10:00
#SBATCH --mem=20GB

# === Configuration ===
BBOX="5000, -10000, 57000, 0"
IN_LABELS="labels_no_overlap.gpkg"
OUT_LABELS="labels_cropped.gpkg"
DATA_DIR="$SCRATCH/GrainSeg/dataset/MWD-1#121"
SUFFIX=""
# =====================

source SLURM/prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/crop_images_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

cp "$DATA_DIR/$IN_LABELS" "$WORK_DIR/"
cp "$DATA_DIR/"*.tif "$WORK_DIR/"

echo "Running cropping script on local storage..."
cd pipelines/data_prep && uv run python -u src/crop_images.py \
    --vector "$WORK_DIR/$IN_LABELS" \
    --out-vector "$WORK_DIR/$OUT_LABELS" \
    --image-dir "$WORK_DIR/" \
    --bbox "$BBOX" \
    --suffix "$SUFFIX"

echo "Copying result back to persistent storage..."
cp "$WORK_DIR/$OUT_LABELS" "$DATA_DIR/$OUT_LABELS"
mkdir -p "$DATA_DIR/cropped"
cp -r "$WORK_DIR/cropped/"* "$DATA_DIR/cropped/"

echo "Done!"
