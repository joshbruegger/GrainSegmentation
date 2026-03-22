sbatch SLURM/evaluate_models_and_plot.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --output-dir /scratch/s4361687/GrainSeg/eval/watershed_val \
  --watershed-tune-root /scratch/s4361687/GrainSeg/runs/watershed_tune

sbatch SLURM/evaluate_models_and_plot.sh \
  --model-dir /scratch/s4361687/GrainSeg/models \
  --image-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --mask-dir /scratch/s4361687/GrainSeg/dataset/train/cropped \
  --output-dir /scratch/s4361687/GrainSeg/eval/cc_val