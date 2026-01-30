#!/bin/bash
#SBATCH --job-name=GrainSegmentation
#SBATCH --output=logs/sam2-%j.log
#SBATCH --mem=60G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=03:40:00

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

uv run python -u src/run_sam2.py \
    --input "$SCRATCH/GrainSeg/dataset/MWD-1#121_s0c0.tif" \
    --output "$SCRATCH/GrainSeg/out" \
    --tile-size 6144 \
    --visualize-probability 0 \
    --no-nms

# uv run python -u src/visualize_rle.py -i "$SCRATCH/GrainSeg/dataset/tiff/MWD-1#121/MWD-1 #121_s0c0x730-58842y2513-53028_ORG.tif" \
#     -r "$SCRATCH/GrainSeg/out/MWD-1 #121_s0c0x730-58842y2513-53028_ORG_rle.json" \
#     -o "$SCRATCH/GrainSeg/out/MWD-1 #121_s0c0x730-58842y2513-53028_ORG.tif_visualization.png" \
#     --tiled