#!/bin/bash
#SBATCH --job-name=RasterToPolygon
#SBATCH --output=logs/raster-to-polygon-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_ROOT/SLURM/prepare_env.sh"

EVAL_DIR="${EVAL_DIR:-}"
INPUT_RASTER="${INPUT_RASTER:-}"
OUTPUT_GPKG="${OUTPUT_GPKG:-}"
OUTPUT_LAYER="${OUTPUT_LAYER:-}"
CLASS_VALUE="${CLASS_VALUE:-1}"
MIN_AREA="${MIN_AREA:-0}"
TMP_ROOT="${TMPDIR:-/tmp}"

function usage {
    echo "Usage:"
    echo "  Single raster: sbatch --export=ALL,INPUT_RASTER=in.png,OUTPUT_GPKG=out.gpkg [optional vars] $0"
    echo "  Eval folder:   sbatch --export=ALL,EVAL_DIR=/path/to/eval [optional vars] $0"
    echo "Optional exports:"
    echo "  EVAL_DIR=<dir>        Evaluation directory containing preds_* folders"
    echo "  OUTPUT_LAYER=<name>   GeoPackage layer name (default: output stem)"
    echo "  CLASS_VALUE=<int>     Semantic class to polygonize (default: 1)"
    echo "  MIN_AREA=<int>        Minimum pixel area per component (default: 0)"
    echo "  NO_FLIP_Y=1           Keep image-space Y coordinates instead of negative-Y"
    exit 1
}

if [[ -n "$EVAL_DIR" && ( -n "$INPUT_RASTER" || -n "$OUTPUT_GPKG" ) ]]; then
    echo "Error: EVAL_DIR mode cannot be combined with INPUT_RASTER/OUTPUT_GPKG."
    usage
fi

if [[ -z "$EVAL_DIR" && ( -z "$INPUT_RASTER" || -z "$OUTPUT_GPKG" ) ]]; then
    echo "Error: INPUT_RASTER and OUTPUT_GPKG must be set for single-raster mode."
    usage
fi

JOB_TOKEN="${SLURM_JOB_ID:-$$}"
WORK_DIR="$TMP_ROOT/raster_to_polygon_$JOB_TOKEN"
mkdir -p "$WORK_DIR"

echo "Syncing data prep environment..."
cd "$REPO_ROOT/src/data_prep"
uv sync

if [[ -n "$EVAL_DIR" ]]; then
    if [[ ! -d "$EVAL_DIR" ]]; then
        echo "Error: Evaluation directory not found: $EVAL_DIR"
        exit 1
    fi

    echo "Copying evaluation directory to fast local storage ($TMP_ROOT)..."
    LOCAL_EVAL_DIR="$WORK_DIR/eval"
    mkdir -p "$LOCAL_EVAL_DIR"
    cp -r "$EVAL_DIR"/. "$LOCAL_EVAL_DIR"/

    shopt -s nullglob
    pred_dirs=("$LOCAL_EVAL_DIR"/preds_*)
    shopt -u nullglob

    if [[ "${#pred_dirs[@]}" -eq 0 ]]; then
        echo "Error: No preds_* directories found in $EVAL_DIR"
        exit 1
    fi

    processed_count=0
    for pred_dir in "${pred_dirs[@]}"; do
        [[ -d "$pred_dir" ]] || continue

        shopt -s nullglob
        raster_paths=("$pred_dir"/*.png "$pred_dir"/*.tif "$pred_dir"/*.tiff)
        shopt -u nullglob

        for raster_path in "${raster_paths[@]}"; do
            [[ -f "$raster_path" ]] || continue

            output_path="${raster_path%.*}.gpkg"
            cmd=(uv run --no-sync python -u raster_to_polygon.py
                --input "$raster_path"
                --output "$output_path"
                --class-value "$CLASS_VALUE"
                --min-area "$MIN_AREA"
            )

            if [[ -n "$OUTPUT_LAYER" ]]; then
                cmd+=(--output-layer "$OUTPUT_LAYER")
            fi

            if [[ "${NO_FLIP_Y:-}" == "1" || "${NO_FLIP_Y:-}" == "true" || "${NO_FLIP_Y:-}" == "True" ]]; then
                cmd+=(--no-flip-y)
            fi

            echo "Converting $(basename "$raster_path") in $(basename "$pred_dir")..."
            printf ' %q' "${cmd[@]}"
            echo
            "${cmd[@]}"

            relative_output="${output_path#"$LOCAL_EVAL_DIR"/}"
            mkdir -p "$(dirname "$EVAL_DIR/$relative_output")"
            cp "$output_path" "$EVAL_DIR/$relative_output"
            processed_count=$((processed_count + 1))
        done
    done

    if [[ "$processed_count" -eq 0 ]]; then
        echo "Error: No prediction rasters found under preds_* directories in $EVAL_DIR"
        exit 1
    fi

    echo "Done! Wrote $processed_count polygon files into preds_* folders under $EVAL_DIR"
    exit 0
fi

if [[ ! -f "$INPUT_RASTER" ]]; then
    echo "Error: Input raster not found: $INPUT_RASTER"
    exit 1
fi

echo "Copying input raster to fast local storage ($TMP_ROOT)..."
INPUT_RASTER_NAME="$(basename "$INPUT_RASTER")"
OUTPUT_GPKG_NAME="$(basename "$OUTPUT_GPKG")"

cp "$INPUT_RASTER" "$WORK_DIR/$INPUT_RASTER_NAME"

CMD=(uv run --no-sync python -u raster_to_polygon.py
    --input "$WORK_DIR/$INPUT_RASTER_NAME"
    --output "$WORK_DIR/$OUTPUT_GPKG_NAME"
    --class-value "$CLASS_VALUE"
    --min-area "$MIN_AREA"
)

if [[ -n "$OUTPUT_LAYER" ]]; then
    CMD+=(--output-layer "$OUTPUT_LAYER")
fi

if [[ "${NO_FLIP_Y:-}" == "1" || "${NO_FLIP_Y:-}" == "true" || "${NO_FLIP_Y:-}" == "True" ]]; then
    CMD+=(--no-flip-y)
fi

echo "Running raster to polygon conversion on local storage..."
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Copying results back to persistent storage..."
mkdir -p "$(dirname "$OUTPUT_GPKG")"
cp "$WORK_DIR/$OUTPUT_GPKG_NAME" "$OUTPUT_GPKG"

echo "Done!"
