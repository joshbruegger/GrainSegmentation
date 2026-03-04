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

source prepare_env.sh


INPUT_PATH="${1:-$SCRATCH/GrainSeg/dataset/MWD-1#121_s0c0.tif}"
INPUT_NAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_NAME%.*}"

uv run python -u src/starting_masks.py \
    --input "$INPUT_PATH" \
    --output "$SCRATCH/GrainSeg/out" \
    --tile-size 6144 \
    --visualize-probability 0 \
    --nms-thresh 0.3 \
    --max-mask-coverage 0.5 \
    --load-mask-cache
    # --save-mask-cache


uv run python -u src/convert_rle_polygon.py rle2json \
    -i "$SCRATCH/GrainSeg/out/${INPUT_STEM}.json" \
    -o "$SCRATCH/GrainSeg/out/${INPUT_STEM}.geojson"