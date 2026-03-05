# Project Reorganization Design: Independent Pipelines with Shared Core

## Overview
The goal is to reorganize the `GrainSegmentation` project into separate pipelines (`data_prep` and `training`) that can maintain independent, potentially conflicting virtual environments and dependencies (e.g., SAM-2 vs. a custom TensorFlow wheel).

## Architecture & Directory Structure
We will adopt an "Independent Projects with Shared `src` Package" approach. This avoids `uv` workspaces, which are explicitly not recommended for conflicting requirements.

```
Projects/GrainSegmentation/
├── pipelines/
│   ├── data_prep/           # SAM-2 stuff
│   │   ├── pyproject.toml   # Contains torch, torchvision, sam-2
│   │   ├── uv.lock          # Independent lockfile
│   │   └── src/             # Scripts (e.g., run_sam.py, crop_images.py, etc.)
│   └── training/            # U-Net stuff
│       ├── pyproject.toml   # Contains custom TF wheel, keras-tuner
│       ├── uv.lock          # Independent lockfile
│       └── src/             # Scripts (e.g., train.py, model.py, etc.)
├── src/                     # Shared utilities package
│   ├── pyproject.toml       # Minimal dependencies required for shared code
│   └── grainsegmentation_core/
│       ├── __init__.py
│       └── utils.py         # Shared logic
└── SLURM/                   # Consolidated SLURM scripts (updated paths)
```

## Dependency Management & Virtual Environments
- **Independent Venvs:** Each pipeline (`data_prep` and `training`) will manage its own `.venv` and `uv.lock`. Commands must be run from within the respective pipeline directory (e.g., `cd pipelines/data_prep && uv run python src/run_sam.py`).
- **Shared Code:** The root `src/` directory will be a standalone Python package (`grainsegmentation_core`). Both pipelines will include it in their `pyproject.toml` as a path dependency:
  ```toml
  [dependencies]
  grainsegmentation-core = { path = "../../src" }
  ```
- **Root Cleanup:** The root `pyproject.toml` and `uv.lock` will be removed to prevent confusion and enforce the separation of environments.

## Migration Plan
1. **Create Structure:** Create `pipelines/data_prep`, `pipelines/training`, and `src/grainsegmentation_core`.
2. **Move Code:**
   - Move `src/preprocess/*` to `pipelines/data_prep/src/`.
   - Move `src/segmentation/*` to `pipelines/training/src/`.
   - Identify any truly shared code and move it to `src/grainsegmentation_core/`.
3. **Update SLURM Scripts:** Move `src/SLURM/` to the root `SLURM/` directory and update all paths within the scripts to point to the new pipeline directories and use `uv run` appropriately.
4. **Create Configurations:** Create `pyproject.toml` files for each pipeline, splitting the current root `pyproject.toml` dependencies appropriately.
5. **Cleanup:** Remove the root `pyproject.toml`, `uv.lock`, and `.venv`.