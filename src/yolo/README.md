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

- Fresh runs load `yolo26x-seg.pt`.
- Default run outputs go under `$SCRATCH/GrainSeg/runs/yolo26-seg/<run-name>/`.
- Default resume checkpoints are resolved as `weights/last.pt` inside that run directory.
- The SLURM wrapper stages the selected YOLO dataset directory into `TMPDIR` so dataset YAMLs with `path: .` still work as-is.
- For custom `--data-yaml` runs, the default run name comes from the YAML stem unless you pass `--run-name`.
- Per the indexed `@Yolo` docs, resume restores the saved Ultralytics training state. This pipeline forwards only the documented safe resume overrides such as `data`, `device`, `imgsz`, `batch`, `workers`, `cache`, and `plots`.
- Resume-time `--epochs` and `--amp` changes are rejected explicitly instead of being accepted and then silently ignored by Ultralytics.
