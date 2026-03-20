# YOLO26 Segmentation Pipeline

This directory contains the Ultralytics-based training pipeline for GrainSegmentation YOLO26 segmentation experiments. It mirrors the existing `src/training` workflow with a thin Python CLI, separate orchestration helpers, and SLURM-first launch scripts.

Implementation choices in this directory follow the indexed `@Yolo` docs for:
- `YOLO(...).train(...)` argument names
- resume behavior through `last.pt`
- standard dataset YAML usage for segmentation runs

## Supported Variants

- `PPL`
- `PPLPPXblend`
- `PPL+PPXblend`
- `PPL+AllPPX`

These variants expect prebuilt YOLO datasets under:

```text
$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/
```

The dataset YAML files remain the contract for channel count and split layout. The pipeline does not regenerate YOLO datasets inside training jobs.

## Local Validation

Use `uv` from this directory:

```bash
uv run python train.py --variant PPL --epochs 1 --project "$SCRATCH/GrainSeg/runs/yolo26-seg-dev"
```

Run built-in Ultralytics hyperparameter tuning with the narrowed `lr0` and `dropout`
search space configured in `pipeline.py`:

```bash
uv run python train.py --variant PPL --tune --tune-epochs 30 --tune-iterations 50 --device 0
```

Save random `train` and `val` visualization samples for a YOLO dataset directory:

```bash
uv run python visualize_dataset.py "$SCRATCH/GrainSeg/dataset/MWD-1#121/yolo/PPL" --output-dir "$SCRATCH/GrainSeg/visualizations/PPL" --num 4
```

You can also override the dataset YAML directly:

```bash
uv run python train.py --data "/path/to/dataset.yaml" --name custom-run
```

Resume the most recent checkpoint for a run:

```bash
uv run python train.py --variant PPL --project "$SCRATCH/GrainSeg/runs/yolo26-seg" --resume
```

Resume an explicit checkpoint:

```bash
uv run python train.py --data "/path/to/dataset.yaml" --name custom-run --resume-checkpoint "/path/to/last.pt"
```

## Evaluation (`evaluate.py`)

Run **Ultralytics validation** (mask mAP and related metrics on the `test:` split from the dataset YAML):

```bash
uv run python evaluate.py --mode val --weights "$SCRATCH/GrainSeg/runs/yolo26-seg/PPL/weights/best.pt" \
  --variant PPL --project "$SCRATCH/GrainSeg/runs/yolo26-seg-eval" --name PPL-test
```

**`sahi`** — SAHI on a **whole held-out test TIFF** (not in the training patch set), with grain polygons from a **GeoPackage in image pixel space** (same conventions as `split_tiff_gpkg_to_yolo`). Reports **COCO mask AP** (AP, AP50, AP75, …) via `pycocotools`, comparable in definition to standard instance-seg benchmarks; this is **not** the same code path as patch `model.val()`. Use **`--output-json`** to save per-image metrics. Whole stacks can be large — prefer high-memory SLURM nodes.

```bash
uv run python evaluate.py --mode sahi --weights "/path/to/best.pt" \
  --test-tiff "/path/to/held-out.tif" --test-gpkg "/path/to/labels.gpkg" \
  --slice-height 1024 --slice-width 1024 --output-json "$SCRATCH/GrainSeg/sahi-eval/test.json"
```

Batch several held-out pairs with a JSON manifest (`[{"test_tiff": "...", "test_gpkg": "..."}, ...]`):

```bash
uv run python evaluate.py --mode sahi --weights "/path/to/best.pt" \
  --manifest pairs.json --output-json all.json
```

Relative paths in the manifest are resolved **relative to the manifest file’s directory** (not the process working directory).

In **`val`** mode, `--name` is always passed to Ultralytics (even if `--project` is omitted). The **`--name` flag is ignored in `sahi` mode**.

In **`sahi`** JSON output, if an image has **no ground-truth instances**, COCO-style summary fields are **`-1`** (undefined), matching `pycocotools` sentinels. Aggregated `mean_*` fields omit `-1` values so empty-GT images do not pull means toward zero. If **no** image contributes a valid value for a given metric, the corresponding **`mean_*` is JSON `null`** (strict JSON; not `NaN`).

Optional **`--sahi-out-dir`** writes SAHI `prediction_visual.png` per whole-image run under that directory.

For full validation or whole-tiff **sahi** runs, use the SLURM wrapper (preferred on the cluster). Quick local smoke tests may use `srun`:

```bash
srun --partition=gpu --gpus=1 uv run python evaluate.py --mode val --weights ./best.pt --variant PPL
sbatch SLURM/eval_yolo26x_seg.sh --mode val --weights "$SCRATCH/GrainSeg/runs/yolo26-seg/PPL/weights/best.pt" --variant PPL
sbatch SLURM/eval_yolo26x_seg.sh --mode sahi --weights "$SCRATCH/GrainSeg/runs/yolo26-seg/PPL/weights/best.pt" \
  --test-tiff "$SCRATCH/path/to/held-out.tif" --test-gpkg "$SCRATCH/path/to/held-out.gpkg" \
  --output-json "$SCRATCH/GrainSeg/sahi-eval/metrics.json"
```

The eval SLURM script stages the dataset into `TMPDIR` like `train_yolo26x_seg.sh` unless you pass `--data-yaml` with an explicit YAML path. **`sahi` does not use the training YAML** — it only needs weights and held-out TIFF/GPKG (or `--manifest`).

## Cluster Usage

Run a single SLURM job directly:

```bash
sbatch SLURM/train_yolo26x_seg.sh --variant PPL --run-name PPL
```

For larger variants, either use the preset submit wrapper or override memory explicitly, for example:

```bash
sbatch --mem=950G SLURM/train_yolo26x_seg.sh --variant "PPL+AllPPX" --run-name "PPL+AllPPX"
```

Submit preset experiment jobs:

```bash
bash SLURM/submit_yolo_experiments.sh --all
```

Resume preset jobs from their latest checkpoints:

```bash
bash SLURM/submit_yolo_experiments.sh --all --resume
```

## Notes

- Fresh runs load `yolo26l-seg.pt`.
- Tune runs use Ultralytics' built-in tuner with the project search space for `lr0` and `dropout`.
- Default run outputs go under `$SCRATCH/GrainSeg/runs/yolo26-seg/<run-name>/`.
- Default resume checkpoints are resolved as `weights/last.pt` inside that run directory.
- The SLURM wrapper stages the selected YOLO dataset directory into `TMPDIR` so dataset YAMLs with `path: .` still work as-is.
- For custom `--data-yaml` runs, the default run name comes from the YAML stem unless you pass `--run-name`.
- Per the indexed `@Yolo` docs, resume restores the saved Ultralytics training state. This pipeline forwards resume overrides such as `data`, `device`, `imgsz`, `batch`, `workers`, `cache`, `plots`, and `epochs` (Ultralytics may still ignore some overrides depending on saved state).
- Resume-time `--amp` / `--no-amp` changes are rejected explicitly so they are not accepted and then silently ignored by Ultralytics.
