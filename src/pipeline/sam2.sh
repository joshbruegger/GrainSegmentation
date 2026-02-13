#!/bin/bash
#SBATCH --job-name=GrainSegmentation
#SBATCH --output=logs/sam2-%j.log
#SBATCH --mem=60G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=04:00:00

echo "Loading modules..."
module purge
module load cuDNN/9.10.1.4-CUDA-12.8.0
module load SciPy-bundle/2025.06-gfbf-2025a
module list

# Ensure uv is in the path
export PATH="$HOME/.local/bin:$PATH"

# build the virtual environment if it doesn't exist
if [ ! -d "$HOME/Projects/GrainSegmentation/.venv" ]; then
    uv venv --system-site-packages
fi

uv sync

INPUT_PATH="${1:-$SCRATCH/GrainSeg/dataset/MWD-1#121_s0c0.tif}"
INPUT_NAME="$(basename "$INPUT_PATH")"
INPUT_STEM="${INPUT_NAME%.*}"

uv run python -u src/run_sam2.py \
    --input "$INPUT_PATH" \
    --output "$SCRATCH/GrainSeg/out" \
    --tile-size 6144 \
    --visualize-probability 0 \
    --nms-thresh 0.3 \
    --max-mask-coverage 0.5 \
    --load-mask-cache
    # --save-mask-cache


    # --merge-overlap \
    # --load-mask-cache \
    # --merge-iom-thresh 0.5




    # --nms-thresh 0.7

    # --no-nms \
    # --merge-overlap \
    # --merge-min-overlap 30

    # --no-nms \
    # --merge-overlap \
    # --merge-min-overlap 0

uv run python -u src/convert_rle_polygon.py rle2json \
    -i "$SCRATCH/GrainSeg/out/${INPUT_STEM}.json" \
    -o "$SCRATCH/GrainSeg/out/${INPUT_STEM}.geojson"

# uv run python -u src/visualize_rle.py -i "$SCRATCH/GrainSeg/dataset/tiff/MWD-1#121/MWD-1 #121_s0c0x730-58842y2513-53028_ORG.tif" \
#     -r "$SCRATCH/GrainSeg/out/MWD-1 #121_s0c0x730-58842y2513-53028_ORG_rle.json" \
#     -o "$SCRATCH/GrainSeg/out/MWD-1 #121_s0c0x730-58842y2513-53028_ORG.tif_visualization.png" \
#     --tiled
