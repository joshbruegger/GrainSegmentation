#!/bin/bash
#SBATCH --job-name=GrainSegmentation_download
#SBATCH --output=download-%j.log
#SBATCH --mem=4GB
#SBATCH --time=00:30:00

source SLURM/prepare_env.sh

cd src/data_prep && uv run python -u download_data.py -o $SCRATCH/GrainSeg/dataset/source