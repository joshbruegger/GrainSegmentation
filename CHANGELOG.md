# CHANGELOG


## v0.3.0 (2026-03-06)

### Features

- Add caching support for dataset preparation in training
  ([`087ac9b`](https://github.com/joshbruegger/GrainSegmentation/commit/087ac9b82e783811963f7e8efd307f53c5829670))

- Enhanced the `build_dataset` function to accept an optional `cache_file` parameter for caching
  datasets to disk. - Updated the training logic to create unique cache file paths for each fold and
  trial when using cross-validation. - Ensured caching is configurable for both training and
  validation datasets to optimize data loading during training.

- Enhance SLURM scripts and training logic
  ([`46df58a`](https://github.com/joshbruegger/GrainSegmentation/commit/46df58a46e2ce3652a236d1e96eebc3f19a3da68))

- Increased memory allocation and CPU resources in SLURM job submission for improved performance. -
  Added functionality to resume training from the latest model checkpoint. - Updated training script
  to support dynamic batch size adjustments based on the number of replicas in sync. - Enabled
  caching in dataset preparation to optimize data loading.


## v0.2.0 (2026-03-06)

### Features

- Add skip-tuning option and adjust training parameters
  ([`1355c83`](https://github.com/joshbruegger/GrainSegmentation/commit/1355c83de770eadc2e87e8447322fd339c7d7e09))

- Introduced `--skip-tuning` argument to bypass hyperparameter tuning and use default settings. -
  Updated SLURM script to extend job time from 10 to 12 hours and modified tuning log directory. -
  Changed patch size from 3008 to 1024 for training. - Enhanced training script to find optimal
  batch size dynamically and adjusted model checkpointing for better tracking.


## v0.1.1 (2026-03-06)

### Bug Fixes

- Distribute uv sync to individual SLURM scripts
  ([`dc0a65d`](https://github.com/joshbruegger/GrainSegmentation/commit/dc0a65dee65711bcb82e73c1f8f00b09cf06c1c0))

### Chores

- Add pyproject.toml and lockfile for data_prep pipeline
  ([`abd2248`](https://github.com/joshbruegger/GrainSegmentation/commit/abd224895c32573284d80f4274ec6f07f4eee978))

Made-with: Cursor

- Add pyproject.toml and lockfile for training pipeline
  ([`7e52941`](https://github.com/joshbruegger/GrainSegmentation/commit/7e529417dfefda91491af9ec94cd337a17bbec6a))

Made-with: Cursor

- Create directory structure for independent pipelines
  ([`ecf2c03`](https://github.com/joshbruegger/GrainSegmentation/commit/ecf2c03a0df54eee167d2db905ecf345dc521d7f))

Made-with: Cursor

- Remove root pyproject.toml and uv.lock to enforce independent pipelines
  ([`f336f0d`](https://github.com/joshbruegger/GrainSegmentation/commit/f336f0dc2a54941dd4790c168eb277ff9355cf78))

Made-with: Cursor

### Documentation

- Add project reorganization design doc
  ([`20295d3`](https://github.com/joshbruegger/GrainSegmentation/commit/20295d3ffdfb03979813509ba3db4ff6aa94e607))

Made-with: Cursor

- Add project reorganization implementation plan
  ([`9696fbc`](https://github.com/joshbruegger/GrainSegmentation/commit/9696fbc145099949561f38b0ac7d68c34c64150d))

Made-with: Cursor

### Refactoring

- Move and update SLURM scripts for new pipeline structure
  ([`d4a637a`](https://github.com/joshbruegger/GrainSegmentation/commit/d4a637ad4a523c56a3d9935c50f3ff7f44b21f9e))

Made-with: Cursor

- Move preprocess code to data_prep pipeline
  ([`6967ce1`](https://github.com/joshbruegger/GrainSegmentation/commit/6967ce140e2b7dbea170412fd04a3b79e379c5b9))

Made-with: Cursor

- Move segmentation code to training pipeline
  ([`f818617`](https://github.com/joshbruegger/GrainSegmentation/commit/f81861706608eface59f7cb797c9c907636bf9c9))

Moved the segmentation module and the UNet training script to the new training pipeline directory
  structure to improve project organization and separate concerns.

Made-with: Cursor


## v0.1.0 (2025-01-29)

### Chores

- Add VSCode configuration for Ruff and formatting
  ([`f44a799`](https://github.com/joshbruegger/GrainSegmentation/commit/f44a7999060c528de34505cf08669ba7c90bec27))

### Features

- Add data download script and update project dependencies
  ([`397dd33`](https://github.com/joshbruegger/GrainSegmentation/commit/397dd33adf92fd21e32dcc368e666a53366f4567))


## v0.0.1 (2025-01-21)

### Bug Fixes

- Readded version in pyproject again
  ([`11608b8`](https://github.com/joshbruegger/GrainSegmentation/commit/11608b8a2d63aa6be9154b475028fccfd7c6b247))

### Chores

- Update versioning configuration and add version module
  ([`fb88f5c`](https://github.com/joshbruegger/GrainSegmentation/commit/fb88f5c224754b6b9b4f08511543845566a58e03))

### Refactoring

- Added src folder and main file
  ([`65640bf`](https://github.com/joshbruegger/GrainSegmentation/commit/65640bf5c750ae9fdd00b054bdc1bdd3eafafcf4))


## v0.0.0 (2025-01-21)
