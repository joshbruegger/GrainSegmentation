#!/bin/bash
#SBATCH --job-name=merge_multichannel_tiff
#SBATCH --output=logs/merge_multichannel_tiff-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:15:00

source SLURM/prepare_env.sh

WORK_DIR="$TMPDIR/merge_multichannel_tiff_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR/cropped"
mkdir -p "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/merged"

echo "Copying PPL and PPX blend to fast local storage ($TMPDIR)..."
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPL.tif" "$WORK_DIR/cropped/"
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPXblend.tif" "$WORK_DIR/cropped/"

echo "Syncing data prep environment..."
cd src/data_prep
uv sync

echo "Merging PPL and PPX blend into a single multichannel TIFF..."
uv run python -u stack_tiff_channels.py "$WORK_DIR/cropped/" "$WORK_DIR/MWD-1#121_PPL+PPXblend.tif"

echo "Copying merged TIFF to persistent storage..."
mv "$WORK_DIR/MWD-1#121_PPL+PPXblend.tif" "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/merged/MWD-1#121_PPL+PPXblend.tif"
rm "$WORK_DIR/cropped/MWD-1#121_PPXblend.tif"

echo "Copying PPX images to fast local storage ($TMPDIR)..."
for i in {1..6}; do
    cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPX${i}.tif" "$WORK_DIR/cropped/"
done

echo "Merging PPL and PPX images into a single multichannel TIFF..."
uv run python -u stack_tiff_channels.py "$WORK_DIR/cropped/" "$WORK_DIR/MWD-1#121_PPL+AllPPX.tif"

mv "$WORK_DIR/MWD-1#121_PPL+AllPPX.tif" "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/merged/MWD-1#121_PPL+AllPPX.tif"

echo "Done!"
