#!/bin/bash
#SBATCH --job-name=GrainSegmentation_download
#SBATCH --output=download-%j.log
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

source prepare_env.sh

uv run python -u src/preprocess/download_data.py -o $SCRATCH/GrainSeg/dataset/source