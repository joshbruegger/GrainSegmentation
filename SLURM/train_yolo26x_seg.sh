#!/bin/bash
#SBATCH --job-name=Train_YOLO
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:2
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

function usage {
    echo "Usage: $0 [--variant <name>] [--data-yaml <path>] [--run-name <name>] [--project <path>] [--resume [checkpoint]] [--epochs <count>] [--device <value>] [--verbose]"
    echo "  --variant <name>: Dataset variant to train (default: PPL)"
    echo "  --data-yaml <path>: Optional explicit dataset YAML path override"
    echo "  --run-name <name>: Stable run name (defaults to the selected variant)"
    echo "  --project <path>: Output project directory (defaults to \$SCRATCH/GrainSeg/runs/yolo26-seg)"
    echo "  --resume [checkpoint]: Resume from the last run checkpoint or an explicit checkpoint path"
    echo "  --epochs <count>: Epoch count forwarded to src/yolo/train.py for fresh runs"
    echo "  --device <value>: Ultralytics device value to forward to src/yolo/train.py"
    echo "  --verbose: Keep shell tracing messages enabled for troubleshooting"
    echo "  For variant-specific memory requests, prefer SLURM/submit_yolo_experiments.sh or override sbatch --mem."
    echo "  Per the indexed @Yolo docs, resume restores saved training state; unsupported resume-time overrides are rejected."
    exit 1
}

VARIANT="PPL"
VARIANT_EXPLICIT=false
DATA_YAML=""
DATA_OVERRIDE=false
RUN_NAME=""
PROJECT_DIR=""
RESUME_MODE=""
EPOCHS=""
DEVICE="0,1"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            VARIANT_EXPLICIT=true
            shift 2
            ;;
        --data-yaml)
            DATA_YAML="$2"
            DATA_OVERRIDE=true
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --project)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -gt 1 && "${2:-}" != -* ]]; then
                RESUME_MODE="$2"
                shift 2
            else
                RESUME_MODE="__LATEST__"
                shift
            fi
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

source "$REPO_ROOT/SLURM/prepare_env.sh"

if [[ -z "$RUN_NAME" ]]; then
    if [[ "$DATA_OVERRIDE" == true && "$VARIANT_EXPLICIT" == false ]]; then
        RUN_NAME="$(basename "${DATA_YAML%.*}")"
    else
        RUN_NAME="$VARIANT"
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$SCRATCH/GrainSeg/runs/yolo26-seg"
fi

case "$VARIANT" in
    PPL)
        DATASET_SUBDIR="PPL"
        YAML_NAME="PPL.yaml"
        ;;
    PPLPPXblend)
        DATASET_SUBDIR="PPLPPXblend"
        YAML_NAME="PPLPPXblend.yaml"
        ;;
    PPL+PPXblend)
        DATASET_SUBDIR="PPL+PPXblend"
        YAML_NAME="PPL_PPXblend.yaml"
        ;;
    PPL+AllPPX)
        DATASET_SUBDIR="PPL+AllPPX"
        YAML_NAME="PPL+AllPPX.yaml"
        ;;
    *)
        echo "Unknown YOLO variant: $VARIANT" >&2
        exit 1
        ;;
esac

if [[ -z "$DATA_YAML" ]]; then
    echo "Copying YOLO dataset to TMPDIR..."
    TMP_YOLO_ROOT="$TMPDIR/yolo"
    TMP_DATASET_DIR="$TMP_YOLO_ROOT/$DATASET_SUBDIR"
    mkdir -p "$TMP_YOLO_ROOT"
    cp -r "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/$DATASET_SUBDIR" "$TMP_YOLO_ROOT/"
    DATA_YAML="$TMP_DATASET_DIR/$YAML_NAME"

    # Root the copied dataset YAML at TMPDIR so Ultralytics resolves images locally.
    python3 - "$DATA_YAML" "$TMP_DATASET_DIR" <<'PY'
from pathlib import Path
import sys

yaml_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
text = yaml_path.read_text(encoding="utf-8")
lines = text.splitlines()

for index, line in enumerate(lines):
    if line.startswith("path:"):
        lines[index] = f"path: {dataset_root}"
        break
else:
    raise SystemExit(f"Dataset YAML missing path entry: {yaml_path}")

trailing_newline = "\n" if text.endswith("\n") else ""
yaml_path.write_text("\n".join(lines) + trailing_newline, encoding="utf-8")
PY
fi

echo "Syncing YOLO environment..."
cd "$REPO_ROOT/src/yolo"
uv sync

TRAIN_CMD=(
    uv run python -u train.py
    --data "$DATA_YAML"
    --name "$RUN_NAME"
    --project "$PROJECT_DIR"
    --device "$DEVICE"
)

if [[ -n "$EPOCHS" ]]; then
    TRAIN_CMD+=(--epochs "$EPOCHS")
fi

if [[ "$DATA_OVERRIDE" == false || "$VARIANT_EXPLICIT" == true ]]; then
    TRAIN_CMD+=(--variant "$VARIANT")
fi

if [[ "$RESUME_MODE" == "__LATEST__" ]]; then
    TRAIN_CMD+=(--resume)
elif [[ -n "$RESUME_MODE" ]]; then
    TRAIN_CMD+=(--resume-checkpoint "$RESUME_MODE")
fi

if [[ "$VERBOSE" == true ]]; then
    echo "Verbose mode enabled."
fi

"${TRAIN_CMD[@]}"
