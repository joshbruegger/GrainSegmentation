#!/bin/bash
#SBATCH --job-name=crop_images
#SBATCH --output=logs/crop_images-%j.log
#SBATCH --time=00:10:00

source prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/crop_images_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

cp $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_no_overlap.gpkg "$WORK_DIR/"
cp $SCRATCH/GrainSeg/dataset/MWD-1#121/MWD-1#121_s0c*.tif "$WORK_DIR/"

echo "Running cropping script on local storage..."
uv run python -u src/preprocess/crop_images.py \
    --vector "$WORK_DIR/labels_no_overlap.gpkg" \
    --out-vector "$WORK_DIR/labels_cropped.gpkg" \
    --image-dir "$WORK_DIR/" \
    --sample MWD-1#121 \
    --bbox "5000, -10000, 57000, 0"

echo "Copying result back to persistent storage..."
cp "$WORK_DIR/labels_cropped.gpkg" $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_cropped.gpkg
mkdir -p $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped
cp -r "$WORK_DIR/cropped/"* $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/

echo "Done!"