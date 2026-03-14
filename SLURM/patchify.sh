#!/bin/bash
#SBATCH --job-name=split_test_val
#SBATCH --output=logs/split_test_val-%j.log
#SBATCH --mem=100GB
#SBATCH --time=01:00:00

source SLURM/prepare_env.sh

WORK_DIR="$TMPDIR/split_test_val_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
cd src/data_prep

echo "Copying input files to fast local storage ($TMPDIR)..."
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/merged/MWD-1#121_PPL+PPXblend.tif" "$WORK_DIR/"
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/merged/MWD-1#121_PPL+AllPPX.tif" "$WORK_DIR/"
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPLPPXblend.tif" "$WORK_DIR/"
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/MWD-1#121_PPL.tif" "$WORK_DIR/"
cp "$SCRATCH/GrainSeg/dataset/MWD-1#121/cropped/labels.gpkg" "$WORK_DIR/"

echo "Running split test val script for all 4 models"
uv run python -u split_tiff_gpkg_to_yolo.py \
    --image "$WORK_DIR/MWD-1#121_PPL.tif" \
    --polygons "$WORK_DIR/labels.gpkg" \
    --output-dir "$WORK_DIR/PPL" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

uv run python -u split_tiff_gpkg_to_yolo.py \
    --image "$WORK_DIR/MWD-1#121_PPLPPXblend.tif" \
    --polygons "$WORK_DIR/labels.gpkg" \
    --output-dir "$WORK_DIR/PPLPPXblend" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

uv run python -u split_tiff_gpkg_to_yolo.py \
    --image "$WORK_DIR/MWD-1#121_PPL+PPXblend.tif" \
    --polygons "$WORK_DIR/labels.gpkg" \
    --output-dir "$WORK_DIR/PPL+PPXblend" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

    uv run python -u split_tiff_gpkg_to_yolo.py \
    --image "$WORK_DIR/MWD-1#121_PPL+AllPPX.tif" \
    --polygons "$WORK_DIR/labels.gpkg" \
    --output-dir "$WORK_DIR/PPL+AllPPX" \
    --patch-size 1024 \
    --patch-overlap 0.5 \
    --tile-size 4096 \
    --validation-fraction 0.2 \
    --random-state 42

mkdir -p "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo"
mv "$WORK_DIR/PPL" "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/PPL"
mv "$WORK_DIR/PPLPPXblend" "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/PPLPPXblend"
mv "$WORK_DIR/PPL+PPXblend" "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/PPL+PPXblend"
mv "$WORK_DIR/PPL+AllPPX" "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/PPL+AllPPX"

echo "Done!"
