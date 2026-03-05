#!/bin/bash

echo "Submitting PPL only (1 input) job..."
sbatch \
    --job-name=GrainSegTrain_PPL \
    --export=ALL,NUM_INPUTS=1,IMAGE_SUFFIXES="_PPL",RUN_NAME="1in_PPL" \
    src/SLURM/train_unet_multi_input.sh

echo "Submitting PPL + PPX Blended (2 inputs) job..."
sbatch \
    --job-name=GrainSegTrain_PPL_Blended \
    --export=ALL,NUM_INPUTS=2,IMAGE_SUFFIXES="_PPL,_PPX_blended",RUN_NAME="2in_PPL_Blended" \
    src/SLURM/train_unet_multi_input.sh

echo "Submitting PPL + All PPX (7 inputs) job..."
sbatch \
    --job-name=GrainSegTrain_PPL_AllPPX \
    --export=ALL,NUM_INPUTS=7,IMAGE_SUFFIXES="_PPL,_PPX1,_PPX2,_PPX3,_PPX4,_PPX5,_PPX6",RUN_NAME="7in_PPL_AllPPX" \
    src/SLURM/train_unet_multi_input.sh

echo "All 3 jobs submitted successfully!"