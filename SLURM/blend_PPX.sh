#!/bin/bash
#SBATCH --job-name=blend_PPX
#SBATCH --output=logs/blend_PPX-%j.log
#SBATCH --mem=20GB
#SBATCH --time=00:05:00

source SLURM/prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/blend_PPX_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR/cropped"

# Copy all the cropped images to the fast TMPDIR
cp -r $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/* "$WORK_DIR/cropped/"

echo "Running blending script on local storage..."
cd pipelines/data_prep && uv run python -u src/blend_tiffs.py \
    "$WORK_DIR/cropped/" \
    "$WORK_DIR/PPX_blended.tif" \
    --exclude "$WORK_DIR/cropped/PPL_crop.tif"

echo "Copying result back to persistent storage..."
cp "$WORK_DIR/PPX_blended.tif" $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/PPX_blended.tif

echo "Done!"
