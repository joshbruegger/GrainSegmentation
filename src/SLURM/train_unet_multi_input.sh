#!/bin/bash
#SBATCH --job-name=GrainSegTrain
#SBATCH --output=logs/unet-train-%j.log
#SBATCH --mem=50G
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=0:05:00

set -euo pipefail

source src/SLURM/prepare_env.sh

uv run python -u src/train_unet_multi_input.py --run-name ${SLURM_JOB_ID:-local} --image-dir $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped --mask-dir $SCRATCH/GrainSeg/dataset/MWD-1#121/cropped --checkpoint models/pretrained/starting_point.keras --output-model $SCRATCH/GrainSeg/models/unet_7in_finetuned.keras --patch-size 3008 --patch-overlap 0.5 --epochs 100 --tune-epochs 20 --image-suffixes _PPL,_PPX1,_PPX2,_PPX3,_PPX4,_PPX5,_PPX6 --mask-ext .tif --mask-stem-suffix _labels