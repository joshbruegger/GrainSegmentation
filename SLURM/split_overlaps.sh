#!/bin/bash
#SBATCH --job-name=split_overlaps
#SBATCH --output=logs/split_overlaps-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source SLURM/prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/split_overlaps_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

cp $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_raw.gpkg "$WORK_DIR/"
cp $SCRATCH/GrainSeg/dataset/MWD-1#121/MWD-1#121_s0c*.tif "$WORK_DIR/"

echo "Running split overlaps script on local storage..."
cd src/data_prep && uv run python -u split_overlaps.py \
    --input "$WORK_DIR/labels_raw.gpkg" \
    --output "$WORK_DIR/labels_no_overlap.gpkg" \
    --min-area 300 \
    --workers 10

echo "Running cropping script on local storage..."
uv run python -u crop_images.py \
    --vector "$WORK_DIR/labels_no_overlap.gpkg" \
    --out-vector "$WORK_DIR/labels_cropped.gpkg" \
    --image-dir "$WORK_DIR/" \
    --sample MWD-1#121 \
    --bbox "5000, -10000, 57000, 0"

echo "Copying results back to persistent storage..."
cp "$WORK_DIR/labels_no_overlap.gpkg" $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_no_overlap.gpkg
cp "$WORK_DIR/labels_cropped.gpkg" $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_cropped.gpkg
mkdir -p $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped
cp -r "$WORK_DIR/cropped/"* $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/

echo "Done!"