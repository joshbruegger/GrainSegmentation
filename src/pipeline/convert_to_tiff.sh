#!/bin/bash
#SBATCH --job-name=GrainSegmentation_convert_to_tiff
#SBATCH --output=convert_to_tiff-%j.log
#SBATCH --mem=100G
#SBATCH --time=04:00:00

echo "Loading modules..."
module purge
module load Python/3.13.1-GCCcore-14.2.0
module list

# Ensure uv is in the path
export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen

uv run python -u src/preprocess/czi_to_tiff.py $SCRATCH/GrainSeg/dataset/source -o $SCRATCH/GrainSeg/dataset/source/tiff -r $SLURM_SUBMIT_DIR/src/data/rois.txt -v --lazy --skip-existing -c zstd