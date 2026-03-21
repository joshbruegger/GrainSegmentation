#!/bin/bash
#SBATCH --job-name=EvalModels
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"

MODEL_DIR=""
IMAGE_DIR=""
MASK_DIR=""
OUTPUT_DIR=""
CONFIG_FILE=""
PPL_IMAGE=""
GT_PATH=""
PATCH_SIZE=1024
STRIDE=512
BATCH_SIZE=1
BOUNDARY_TOLERANCE=2.0
MASK_EXT=".tif"
MASK_STEM_SUFFIX="_labels"
# If set, auto-resolve watershed_best_*.json under <root>/<variant_subdir>/ per model stem.
WATERSHED_TUNE_ROOT=""

function usage {
    echo "Usage: $0 --model-dir <dir> --image-dir <dir> --mask-dir <dir> --output-dir <dir> [options]"
    echo "  --model-dir <dir>         Directory containing .keras models to evaluate"
    echo "  --image-dir <dir>         Directory containing evaluation images"
    echo "  --mask-dir <dir>          Directory containing raster masks"
    echo "  --output-dir <dir>        Directory for JSONs, predictions, and plots"
    echo "  --config-file <path>      Optional TSV: label, model, num_inputs, suffix_csv [, watershed_json]"
    echo "  --ppl-image <path>        Optional PPL image to use for overlay generation"
    echo "  --gt-path <path>          Optional mask path to use for overlay generation"
    echo "  --patch-size <int>        Evaluation patch size (default: 1024)"
    echo "  --stride <int>            Evaluation stride (default: 512)"
    echo "  --batch-size <int>        Evaluation batch size (default: 1)"
    echo "  --boundary-tolerance <f>  Boundary tolerance passed to evaluate.py (default: 2.0)"
    echo "  --mask-ext <ext>          Optional mask extension override (default: .tif)"
    echo "  --mask-stem-suffix <s>    Optional suffix before the mask extension (default: '_labels')"
    echo "  --watershed-tune-root <d> Optional: directory containing per-variant watershed_tune subdirs"
    echo "                            (e.g. .../runs/watershed_tune). Picks latest watershed_best_*.json."
    echo
    echo "Config TSV columns: label, model, num_inputs, suffix_csv [, optional_watershed_json_path]"
    echo "If --config-file is omitted, known model naming presets are inferred from filenames."
    exit 1
}

function require_file {
    local path="$1"
    local message="$2"
    if [ ! -f "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

function require_dir {
    local path="$1"
    local message="$2"
    if [ ! -d "$path" ]; then
        echo "$message: $path"
        exit 1
    fi
}

function strip_prefix {
    local value="$1"
    local prefix="$2"
    if [[ "$value" == "$prefix"* ]]; then
        printf '%s\n' "${value#"$prefix"}"
    else
        printf '%s\n' "$value"
    fi
}

function stage_optional_path {
    local original="$1"
    local original_root="$2"
    local local_root="$3"

    if [ -z "$original" ]; then
        printf '\n'
        return
    fi

    if [[ "$original" == "$original_root"/* ]]; then
        local relative="${original#"$original_root"/}"
        printf '%s\n' "$local_root/$relative"
        return
    fi

    printf '%s\n' "$original"
}

function resolve_config_model_path {
    local model_ref="$1"

    if [[ "$model_ref" = /* ]]; then
        local staged_path="$LOCAL_MODEL_DIR/$(basename "$model_ref")"
        if [ ! -f "$staged_path" ]; then
            cp "$model_ref" "$staged_path"
        fi
        printf '%s\n' "$staged_path"
        return
    fi

    printf '%s\n' "$LOCAL_MODEL_DIR/$model_ref"
}

function infer_model_config {
    local model_path="$1"
    local model_file
    local model_stem
    local label

    model_file="$(basename "$model_path")"
    model_stem="${model_file%.keras}"
    label="$(strip_prefix "$model_stem" "unet_finetuned_")"

    if [[ "$model_stem" == *"PPL+AllPPX"* ]]; then
        printf '%s\t7\t_PPL,_PPX1,_PPX2,_PPX3,_PPX4,_PPX5,_PPX6\n' "$label"
        return 0
    fi
    if [[ "$model_stem" == *"PPL+PPXblend"* ]]; then
        printf '%s\t2\t_PPL,_PPXblend\n' "$label"
        return 0
    fi
    if [[ "$model_stem" == *"PPLPPXblend"* ]]; then
        printf '%s\t1\t_PPLPPXblend\n' "$label"
        return 0
    fi
    if [[ "$model_stem" == *"PPL"* ]]; then
        printf '%s\t1\t_PPL\n' "$label"
        return 0
    fi

    return 1
}

# Subdir names match SLURM/submit_tune_watershed_variants.sh --output-dir basename.
function infer_watershed_tune_subdir_from_stem {
    local model_stem="$1"

    if [[ "$model_stem" == *"PPL+AllPPX"* ]]; then
        printf '%s\n' "PPL_AllPPX"
        return 0
    fi
    if [[ "$model_stem" == *"PPL+PPXblend"* ]]; then
        printf '%s\n' "PPL_PlusPPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPLPPXblend"* ]]; then
        printf '%s\n' "PPLPPXblend"
        return 0
    fi
    if [[ "$model_stem" == *"PPL"* ]]; then
        printf '%s\n' "PPL"
        return 0
    fi

    return 1
}

function pick_latest_watershed_best_json {
    local dir="$1"
    shopt -s nullglob
    local matches=("$dir"/watershed_best_*.json)
    shopt -u nullglob

    if [ "${#matches[@]}" -eq 0 ]; then
        echo "No watershed_best_*.json files in: $dir" >&2
        return 1
    fi

    local newest=""
    local newest_mtime=0
    for f in "${matches[@]}"; do
        local m
        m="$(stat -c '%Y' "$f" 2>/dev/null || stat -f '%m' "$f")"
        if [ "$m" -gt "$newest_mtime" ]; then
            newest_mtime="$m"
            newest="$f"
        fi
    done
    printf '%s\n' "$newest"
}

function resolve_watershed_json_for_model {
    local model_path="$1"
    local explicit_json="${2:-}"

    if [ -n "$explicit_json" ]; then
        if [[ "$explicit_json" = /* ]]; then
            printf '%s\n' "$explicit_json"
            return 0
        fi
        printf '%s\n' "$MODEL_DIR/$explicit_json"
        return 0
    fi

    if [ -z "$WATERSHED_TUNE_ROOT" ]; then
        printf '\n'
        return 0
    fi

    local model_file model_stem subdir variant_dir
    model_file="$(basename "$model_path")"
    model_stem="${model_file%.keras}"

    if ! subdir="$(infer_watershed_tune_subdir_from_stem "$model_stem")"; then
        echo "Cannot infer watershed tune subdir for model stem: $model_stem" >&2
        return 1
    fi

    variant_dir="$WATERSHED_TUNE_ROOT/$subdir"
    if [ ! -d "$variant_dir" ]; then
        echo "Watershed tune variant directory not found: $variant_dir" >&2
        return 1
    fi

    pick_latest_watershed_best_json "$variant_dir"
}

function find_default_ppl_image {
    shopt -s nullglob
    local matches=("$LOCAL_IMAGE_DIR"/*_PPL.*)
    shopt -u nullglob

    if [ "${#matches[@]}" -eq 0 ]; then
        echo "Unable to infer --ppl-image; no *_PPL.* file found in $IMAGE_DIR"
        exit 1
    fi

    printf '%s\n' "${matches[0]}"
}

function infer_overlay_sample_id {
    local ppl_path="$1"
    local base_name
    local stem

    base_name="$(basename "$ppl_path")"
    stem="${base_name%.*}"

    if [[ "$stem" != *_PPL ]]; then
        echo "Unable to infer overlay sample id from PPL image: $ppl_path"
        exit 1
    fi

    printf '%s\n' "${stem%_PPL}"
}

function find_mask_for_sample {
    local sample_id="$1"
    local candidate=""

    if [ -n "$MASK_EXT" ]; then
        candidate="$LOCAL_MASK_DIR/${sample_id}${MASK_STEM_SUFFIX}${MASK_EXT}"
        require_file "$candidate" "Mask not found for overlay sample"
        printf '%s\n' "$candidate"
        return
    fi

    local ext=""
    for ext in .png .tif .tiff .jpg .jpeg; do
        candidate="$LOCAL_MASK_DIR/${sample_id}${MASK_STEM_SUFFIX}${ext}"
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "Unable to infer ground-truth mask for overlay sample: $sample_id"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --mask-dir)
            MASK_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ppl-image)
            PPL_IMAGE="$2"
            shift 2
            ;;
        --gt-path)
            GT_PATH="$2"
            shift 2
            ;;
        --patch-size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --boundary-tolerance)
            BOUNDARY_TOLERANCE="$2"
            shift 2
            ;;
        --mask-ext)
            MASK_EXT="$2"
            shift 2
            ;;
        --mask-stem-suffix)
            MASK_STEM_SUFFIX="$2"
            shift 2
            ;;
        --watershed-tune-root)
            WATERSHED_TUNE_ROOT="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL_DIR" ] || [ -z "$IMAGE_DIR" ] || [ -z "$MASK_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
fi

require_dir "$MODEL_DIR" "Model directory not found"
require_dir "$IMAGE_DIR" "Image directory not found"
require_dir "$MASK_DIR" "Mask directory not found"
if [ -n "$CONFIG_FILE" ]; then
    require_file "$CONFIG_FILE" "Config file not found"
fi

mkdir -p "$OUTPUT_DIR"

source "$REPO_ROOT/SLURM/prepare_env.sh"
export TF_CPP_MIN_LOG_LEVEL=2

WORK_DIR="$TMPDIR/eval_models_${SLURM_JOB_ID:-$$}"
LOCAL_MODEL_DIR="$WORK_DIR/models"
LOCAL_IMAGE_DIR="$WORK_DIR/images"
LOCAL_MASK_DIR="$WORK_DIR/masks"
mkdir -p "$LOCAL_MODEL_DIR" "$LOCAL_IMAGE_DIR" "$LOCAL_MASK_DIR"

echo "Copying models and dataset to TMPDIR..."
cp -r "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
cp -r "$IMAGE_DIR"/. "$LOCAL_IMAGE_DIR"/
cp -r "$MASK_DIR"/. "$LOCAL_MASK_DIR"/

cd "$REPO_ROOT/src/training"
echo "Syncing evaluation environment..."
uv sync

WHEEL_PATH="$SCRATCH/GrainSeg/wheels/$TF_WHEEL_NAME"
require_file "$WHEEL_PATH" "TensorFlow wheel not found"
echo "Installing TensorFlow wheel..."
uv pip install nvidia-cudnn-cu12~=9.0 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12~=12.8.0 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 "$WHEEL_PATH"

MODEL_LABELS=()
MODEL_PATHS=()
MODEL_NUM_INPUTS=()
MODEL_SUFFIXES=()
MODEL_WATERSHED_JSONS=()
JSON_FILES=()
PRED_PATHS=()

if [ -n "$CONFIG_FILE" ]; then
    while IFS=$'\t' read -r label model_ref num_inputs suffix_csv watershed_json_opt || [ -n "$label" ]; do
        if [ -z "${label// }" ] || [[ "$label" == \#* ]]; then
            continue
        fi

        if [ -z "${model_ref:-}" ] || [ -z "${num_inputs:-}" ] || [ -z "${suffix_csv:-}" ]; then
            echo "Invalid config row in $CONFIG_FILE: $label"
            exit 1
        fi

        local_model_path="$(resolve_config_model_path "$model_ref")"
        require_file "$local_model_path" "Configured model not found"

        MODEL_LABELS+=("$label")
        MODEL_PATHS+=("$local_model_path")
        MODEL_NUM_INPUTS+=("$num_inputs")
        MODEL_SUFFIXES+=("$suffix_csv")
        MODEL_WATERSHED_JSONS+=("${watershed_json_opt:-}")
    done < "$CONFIG_FILE"
else
    shopt -s nullglob globstar
    local_models=("$LOCAL_MODEL_DIR"/**/*.keras)
    shopt -u nullglob globstar

    if [ "${#local_models[@]}" -eq 0 ]; then
        echo "No .keras models found in $MODEL_DIR"
        exit 1
    fi

    for model_path in "${local_models[@]}"; do
        if ! inferred="$(infer_model_config "$model_path")"; then
            echo "Unable to infer config for model; use --config-file instead: $model_path"
            exit 1
        fi

        IFS=$'\t' read -r label num_inputs suffix_csv <<< "$inferred"
        MODEL_LABELS+=("$label")
        MODEL_PATHS+=("$model_path")
        MODEL_NUM_INPUTS+=("$num_inputs")
        MODEL_SUFFIXES+=("$suffix_csv")
        MODEL_WATERSHED_JSONS+=("")
    done
fi

if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
    echo "No models configured for evaluation."
    exit 1
fi

WATERSHED_JSON_HELPER="$REPO_ROOT/src/evaluation/watershed_json_to_eval_args.py"

echo "Running evaluations..."
for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_file="$(basename "$model_path")"
    model_stem="${model_file%.keras}"
    pred_dir="$OUTPUT_DIR/preds_${model_stem}"
    json_path="$OUTPUT_DIR/${model_stem}.json"
    suffix_csv="${MODEL_SUFFIXES[$i]}"
    IFS=',' read -r -a suffix_array <<< "$suffix_csv"

    mkdir -p "$pred_dir"

    eval_cmd=(
        uv run --no-sync python -u ../evaluation/evaluate.py
        --model-path "$model_path"
        --image-dir "$LOCAL_IMAGE_DIR"
        --mask-dir "$LOCAL_MASK_DIR"
        --output-json "$json_path"
        --save-predictions-dir "$pred_dir"
        --num-inputs "${MODEL_NUM_INPUTS[$i]}"
        --image-suffixes
        "${suffix_array[@]}"
        --patch-size "$PATCH_SIZE"
        --stride "$STRIDE"
        --batch-size "$BATCH_SIZE"
        --boundary-tolerance "$BOUNDARY_TOLERANCE"
        --mask-stem-suffix "$MASK_STEM_SUFFIX"
    )

    if [ -n "$MASK_EXT" ]; then
        eval_cmd+=(--mask-ext "$MASK_EXT")
    fi

    explicit_ws="${MODEL_WATERSHED_JSONS[$i]:-}"
    resolved_ws_json=""
    if resolved_ws_json="$(resolve_watershed_json_for_model "$model_path" "$explicit_ws")"; then
        :
    else
        exit 1
    fi

    if [ -n "$resolved_ws_json" ]; then
        require_file "$resolved_ws_json" "Watershed tuning JSON not found"
        if [ ! -f "$WATERSHED_JSON_HELPER" ]; then
            echo "Missing helper script: $WATERSHED_JSON_HELPER" >&2
            exit 1
        fi
        mapfile -t _watershed_eval_args < <(python3 "$WATERSHED_JSON_HELPER" "$resolved_ws_json")
        eval_cmd+=("${_watershed_eval_args[@]}")
    fi

    echo "Evaluating ${MODEL_LABELS[$i]} from $model_file"
    "${eval_cmd[@]}"

    JSON_FILES+=("$json_path")
    PRED_PATHS+=("$pred_dir")
done

LOCAL_PPL_IMAGE="$(stage_optional_path "$PPL_IMAGE" "$IMAGE_DIR" "$LOCAL_IMAGE_DIR")"
if [ -z "$LOCAL_PPL_IMAGE" ]; then
    LOCAL_PPL_IMAGE="$(find_default_ppl_image)"
fi
require_file "$LOCAL_PPL_IMAGE" "Overlay PPL image not found"

OVERLAY_SAMPLE_ID="$(infer_overlay_sample_id "$LOCAL_PPL_IMAGE")"
LOCAL_GT_PATH="$(stage_optional_path "$GT_PATH" "$MASK_DIR" "$LOCAL_MASK_DIR")"
if [ -z "$LOCAL_GT_PATH" ]; then
    LOCAL_GT_PATH="$(find_mask_for_sample "$OVERLAY_SAMPLE_ID")"
fi
require_file "$LOCAL_GT_PATH" "Overlay ground-truth mask not found"

OVERLAY_PRED_PATHS=()
for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_stem="$(basename "${model_path%.keras}")"
    pred_path="$OUTPUT_DIR/preds_${model_stem}/${OVERLAY_SAMPLE_ID}_pred.png"
    require_file "$pred_path" "Overlay prediction not found"
    OVERLAY_PRED_PATHS+=("$pred_path")
done

echo "Generating comparison plots..."
plot_cmd=(
    uv run --no-sync python -u ../evaluation/plot_results.py
    --json-files
    "${JSON_FILES[@]}"
    --labels
    "${MODEL_LABELS[@]}"
    --output-plot "$OUTPUT_DIR/quantitative_plot.png"
)
"${plot_cmd[@]}"

overlay_cmd=(
    uv run --no-sync python -u ../evaluation/plot_results.py
    --image-path "$LOCAL_PPL_IMAGE"
    --gt-path "$LOCAL_GT_PATH"
    --pred-paths
    "${OVERLAY_PRED_PATHS[@]}"
    --labels
    "${MODEL_LABELS[@]}"
    --output-overlay "$OUTPUT_DIR/overlay.png"
)
"${overlay_cmd[@]}"

echo "Saved evaluation outputs to $OUTPUT_DIR"
