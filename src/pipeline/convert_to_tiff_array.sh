#!/bin/bash
#SBATCH --job-name=GrainSeg_convert_array
#SBATCH --output=convert_to_tiff-%A_%a.log
#SBATCH --array=0-9
#SBATCH --mem=100GB
#SBATCH --time=10:00:00

echo "Loading modules..."
module purge
module load Python/3.13.1-GCCcore-14.2.0
module list

export PATH="$HOME/.local/bin:$PATH"

uv sync --frozen

INPUT_DIR="$SCRATCH/GrainSeg/dataset/source"
OUT_DIR="$SCRATCH/GrainSeg/dataset/source/tiff"
ROI_FILE="$SLURM_SUBMIT_DIR/src/data/rois.txt"

mapfile -t FILES < <(ls -1 "$INPUT_DIR"/*.czi | sort)
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILE="${FILES[$IDX]}"

echo "[array] Index=$IDX File=$FILE"

uv run python -u src/preprocess/czi_to_tiff.py "$INPUT_DIR" \
  -o "$OUT_DIR" -r "$ROI_FILE" -v --lazy --skip-existing -c none -t 4096 4096 \
  --only "^$(basename "$FILE" | sed -e 's/[].[^$*\/?+{}()|\^]/\\&/g')$"


