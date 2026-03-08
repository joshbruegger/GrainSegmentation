#!/bin/bash
#SBATCH --job-name=blend_PPX
#SBATCH --output=logs/blend_PPX-%j.log
#SBATCH --mem=20GB
#SBATCH --time=00:05:00

source SLURM/prepare_env.sh

echo "Copying input files to fast local storage ($TMPDIR)..."
WORK_DIR="$TMPDIR/blend_PPX_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR/cropped"

# Copy only MWD-1#121_PPX1.tif to MWD-1#121_PPX6.tif to the fast TMPDIR
for i in {1..6}; do
    cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPX${i}.tif" "$WORK_DIR/cropped/"
done
# cp -r $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/* "$WORK_DIR/cropped/"

echo "Syncing data prep environment..."
cd src/data_prep
uv sync

echo "Running blending script for PPX"
uv run --no-sync python -u blend_tiffs.py \
    "$WORK_DIR/cropped/" \
    "$WORK_DIR/MWD-1#121_PPXblend.tif"

mv "$WORK_DIR/MWD-1#121_PPXblend.tif" $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPXblend.tif

echo "Running blending script for PPL+PPX"
# copy PPL crop to the fast TMPDIR
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPL.tif" "$WORK_DIR/cropped/"

uv run --no-sync python -u blend_tiffs.py \
    "$WORK_DIR/cropped/" \
    "$WORK_DIR/121_PPLPPXblend.tif"

mv "$WORK_DIR/121_PPLPPXblend.tif" $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPLPPXblend.tif

echo "Done!"
