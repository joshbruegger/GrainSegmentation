#!/bin/bash
#SBATCH --job-name=GrainSegmentation
#SBATCH --output=logs/split_overlaps-%j.log
#SBATCH --mem=50GB
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source prepare_env.sh


uv run python -u src/preprocess/split_overlaps.py \
    --input $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_raw.gpkg \
    --output $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_no_overlap.gpkg \
    --min-area 300 \
    --workers 10

uv run python -u src/preprocess/crop_images.py \
    --vector $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_no_overlap.gpkg \
    --out-vector $SCRATCH/GrainSeg/dataset/MWD-1#121/labels_cropped.gpkg \
    --image-dir $SCRATCH/GrainSeg/dataset/MWD-1#121/ \
    --sample MWD-1#121 \
    --bbox "5000, -10000, 57000, 0"