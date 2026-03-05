#!/bin/bash
#SBATCH --job-name=GrainSegmentation_download
#SBATCH --output=download-%j.log
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

source SLURM/prepare_env.sh

cd pipelines/data_prep && uv run python -u src/download_data.py -o $SCRATCH/GrainSeg/dataset/source