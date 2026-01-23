#!/bin/bash
#SBATCH --job-name=GrainSegmentation_download
#SBATCH --output=download-%j.log
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

echo "Loading modules..."
module purge
module load Python/3.13.1-GCCcore-14.2.0
module list

# Ensure uv is in the path
export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen

uv run python -u src/preprocess/download_data.py -o $SCRATCH/GrainSeg/dataset/source