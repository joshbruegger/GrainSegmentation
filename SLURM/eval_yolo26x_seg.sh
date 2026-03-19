#!/bin/bash
#SBATCH --job-name=Eval_YOLO
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

function usage {
    echo "Usage: $0 --mode <val|benchmark|sahi> --weights <path-to.pt> [options]"
    echo "  --mode: val (Ultralytics validator), benchmark (export/runtime), or sahi (tiled val inference)"
    echo "  --weights: Checkpoint path, e.g. .../weights/best.pt"
    echo "  --variant <name>: Dataset variant (default: PPL)"
    echo "  --data-yaml <path>: Override dataset YAML (skips TMPDIR dataset staging when set explicitly)"
    echo "  --device <value>: Ultralytics device (default: 0)"
    echo "  --imgsz <int>: Input size (default: 1024)"
    echo "  --batch <int>: Val batch size (default: 16)"
    echo "  --project <path>: Val project dir for artifacts"
    echo "  --name <str>: Val run name (default: eval)"
    echo "  --format <str>: Benchmark export format only (e.g. onnx; empty = all)"
    echo "  --sahi-out-dir <path>: Required for mode=sahi; output root for prediction visuals"
    echo "  --slice-height/--slice-width: SAHI slice size (default: 1024)"
    echo "  --overlap-height-ratio / --overlap-width-ratio: SAHI overlap (default: 0.2)"
    echo "  --max-images <n>: SAHI only — cap number of val images"
    echo "  --verbose: Bash xtrace"
    echo "  Stage YOLO dataset into TMPDIR like train_yolo26x_seg.sh unless --data-yaml is set."
    exit 1
}

MODE=""
WEIGHTS=""
VARIANT="PPL"
DATA_YAML=""
DEVICE="0"
IMGSZ=1024
BATCH=16
PROJECT_DIR=""
RUN_NAME="eval"
FORMAT=""
SAHI_OUT=""
SLICE_H=1024
SLICE_W=1024
OV_H=0.2
OV_W=0.2
MAX_IMAGES=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --data-yaml)
            DATA_YAML="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --project)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --sahi-out-dir)
            SAHI_OUT="$2"
            shift 2
            ;;
        --slice-height)
            SLICE_H="$2"
            shift 2
            ;;
        --slice-width)
            SLICE_W="$2"
            shift 2
            ;;
        --overlap-height-ratio)
            OV_H="$2"
            shift 2
            ;;
        --overlap-width-ratio)
            OV_W="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
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

if [[ -z "$MODE" || -z "$WEIGHTS" ]]; then
    echo "error: --mode and --weights are required" >&2
    usage
fi

if [[ "$MODE" == "sahi" && -z "$SAHI_OUT" ]]; then
    echo "error: --sahi-out-dir is required for mode=sahi" >&2
    usage
fi

if [[ "$VERBOSE" == true ]]; then
    set -x
fi

source "$REPO_ROOT/SLURM/prepare_env.sh"

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
    echo "Staging YOLO dataset to TMPDIR for evaluation..."
    TMP_YOLO_ROOT="$TMPDIR/yolo"
    TMP_DATASET_DIR="$TMP_YOLO_ROOT/$DATASET_SUBDIR"
    mkdir -p "$TMP_YOLO_ROOT"
    cp -r "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/$DATASET_SUBDIR" "$TMP_YOLO_ROOT/"
    DATA_YAML="$TMP_DATASET_DIR/$YAML_NAME"

    uv run python - "$DATA_YAML" "$TMP_DATASET_DIR" <<'PY'
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

export YOLO_DISABLE_TQDM=True

EVAL_CMD=(
    uv run python -u evaluate.py
    --mode "$MODE"
    --weights "$WEIGHTS"
    --data "$DATA_YAML"
    --device "$DEVICE"
    --imgsz "$IMGSZ"
    --batch "$BATCH"
    --name "$RUN_NAME"
)

if [[ -n "$PROJECT_DIR" ]]; then
    EVAL_CMD+=(--project "$PROJECT_DIR")
fi

if [[ -n "$FORMAT" ]]; then
    EVAL_CMD+=(--format "$FORMAT")
fi

if [[ "$MODE" == "sahi" ]]; then
    EVAL_CMD+=(
        --sahi-out-dir "$SAHI_OUT"
        --slice-height "$SLICE_H"
        --slice-width "$SLICE_W"
        --overlap-height-ratio "$OV_H"
        --overlap-width-ratio "$OV_W"
    )
    if [[ -n "$MAX_IMAGES" ]]; then
        EVAL_CMD+=(--max-images "$MAX_IMAGES")
    fi
fi

"${EVAL_CMD[@]}"
