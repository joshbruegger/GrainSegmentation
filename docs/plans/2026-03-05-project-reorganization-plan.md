# Project Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the repository into independent pipelines (`data_prep` and `training`) with a shared `src` package, allowing for conflicting dependencies and separate virtual environments.

**Architecture:** We will create `pipelines/data_prep`, `pipelines/training`, and `src/grainsegmentation_core`. We will move existing code into these directories, create separate `pyproject.toml` files for each pipeline, and update SLURM scripts to point to the new paths and use `uv run`.

**Tech Stack:** Python, `uv` (for dependency management), SLURM

---

### Task 1: Create Directory Structure and Shared Core

**Files:**
- Create: `pipelines/data_prep/src/`
- Create: `pipelines/training/src/`
- Create: `src/grainsegmentation_core/__init__.py`
- Create: `src/pyproject.toml`
- Create: `SLURM/`

**Step 1: Create directories**

```bash
mkdir -p pipelines/data_prep/src pipelines/training/src src/grainsegmentation_core SLURM
```

**Step 2: Create shared core pyproject.toml**

```toml
# src/pyproject.toml
[project]
name = "grainsegmentation-core"
version = "0.1.0"
description = "Shared utilities for GrainSegmentation"
requires-python = ">=3.12"
dependencies = []
```

**Step 3: Create shared core __init__.py**

```python
# src/grainsegmentation_core/__init__.py
"""Shared utilities for GrainSegmentation."""
```

**Step 4: Commit**

```bash
git add pipelines/ src/grainsegmentation_core/ src/pyproject.toml SLURM/
git commit -m "chore: create directory structure for independent pipelines"
```

---

### Task 2: Move Code to Data Prep Pipeline

**Files:**
- Move: `src/preprocess/*` to `pipelines/data_prep/src/`
- Delete: `src/preprocess/`

**Step 1: Move files**

```bash
mv src/preprocess/* pipelines/data_prep/src/
rmdir src/preprocess
```

**Step 2: Commit**

```bash
git add src/preprocess pipelines/data_prep/src/
git commit -m "refactor: move preprocess code to data_prep pipeline"
```

---

### Task 3: Move Code to Training Pipeline

**Files:**
- Move: `src/segmentation/*` to `pipelines/training/src/`
- Move: `src/train_unet_multi_input.py` to `pipelines/training/src/`
- Delete: `src/segmentation/`

**Step 1: Move files**

```bash
mv src/segmentation/* pipelines/training/src/
mv src/train_unet_multi_input.py pipelines/training/src/
rmdir src/segmentation
```

**Step 2: Commit**

```bash
git add src/segmentation src/train_unet_multi_input.py pipelines/training/src/
git commit -m "refactor: move segmentation code to training pipeline"
```

---

### Task 4: Create Data Prep Pipeline Configuration

**Files:**
- Create: `pipelines/data_prep/pyproject.toml`

**Step 1: Create pyproject.toml for data_prep**

```toml
# pipelines/data_prep/pyproject.toml
[project]
name = "data-prep-pipeline"
version = "0.1.0"
description = "Data preparation pipeline for GrainSegmentation"
requires-python = ">=3.12"
dependencies = [
    "gdown>=5.2.0",
    "sam-2",
    "torch>=2.9.1",
    "torchvision>=0.24.1",
    "opencv-python>=4.12.0.88",
    "large-image>=1.33.5",
    "large-image-source-tiff>=1.33.5",
    "large-image-source-tifffile>=1.33.5",
    "tqdm>=4.67.1",
    "geopandas>=1.1.2",
    "pygeoops>=0.6.0",
    "rasterio>=1.5.0",
    "shapelysmooth>=0.2.1",
    "grainsegmentation-core",
]

[tool.uv.sources]
sam-2 = { git = "https://github.com/facebookresearch/sam2.git" }
grainsegmentation-core = { path = "../../src" }
```

**Step 2: Generate lockfile**

```bash
cd pipelines/data_prep && uv lock
```

**Step 3: Commit**

```bash
git add pipelines/data_prep/pyproject.toml pipelines/data_prep/uv.lock
git commit -m "chore: add pyproject.toml and lockfile for data_prep pipeline"
```

---

### Task 5: Create Training Pipeline Configuration

**Files:**
- Create: `pipelines/training/pyproject.toml`

**Step 1: Create pyproject.toml for training**

```toml
# pipelines/training/pyproject.toml
[project]
name = "training-pipeline"
version = "0.1.0"
description = "Training pipeline for GrainSegmentation"
requires-python = ">=3.12"
dependencies = [
    "matplotlib>=3.10.7",
    "keras-tuner>=1.4.8",
    "scipy>=1.17.1",
    "scikit-learn>=1.8.0",
    "grainsegmentation-core",
]

[tool.uv.sources]
grainsegmentation-core = { path = "../../src" }
```

**Step 2: Generate lockfile**

```bash
cd pipelines/training && uv lock
```

**Step 3: Commit**

```bash
git add pipelines/training/pyproject.toml pipelines/training/uv.lock
git commit -m "chore: add pyproject.toml and lockfile for training pipeline"
```

---

### Task 6: Move and Update SLURM Scripts

**Files:**
- Move: `src/SLURM/*` to `SLURM/`
- Delete: `src/SLURM/`
- Modify: `SLURM/*.sh` (Update paths to point to new pipeline directories)

**Step 1: Move files**

```bash
mv src/SLURM/* SLURM/
rmdir src/SLURM
```

**Step 2: Update paths in SLURM scripts**
We need to update the paths in the SLURM scripts to point to the new pipeline directories and use `uv run` appropriately.
For example, `python src/preprocess/crop_images.py` becomes `cd pipelines/data_prep && uv run python src/crop_images.py`.

*(Note: The exact sed commands or manual edits will be performed during execution based on the contents of each script.)*

**Step 3: Commit**

```bash
git add src/SLURM SLURM/
git commit -m "refactor: move and update SLURM scripts for new pipeline structure"
```

---

### Task 7: Cleanup Root Directory

**Files:**
- Delete: `pyproject.toml`
- Delete: `uv.lock`
- Delete: `.venv/` (if it exists)
- Delete: `src/version.py` (Move to core if needed, but for now, we'll remove root config)

**Step 1: Remove root config files**

```bash
rm pyproject.toml uv.lock
rm -rf .venv
```

**Step 2: Commit**

```bash
git rm pyproject.toml uv.lock
git commit -m "chore: remove root pyproject.toml and uv.lock to enforce independent pipelines"
```
