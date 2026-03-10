# CHANGELOG


## v0.9.0 (2026-03-10)

### Features

- Add frozen CV epoch selection
  ([`f030cf9`](https://github.com/joshbruegger/GrainSegmentation/commit/f030cf9c7fddd96b887d8554e965bd306ce6a275))


## v0.8.0 (2026-03-10)

### Bug Fixes

- Make training restarts predictable
  ([`9002951`](https://github.com/joshbruegger/GrainSegmentation/commit/90029513338fc24dfa5ec1822911f6bfbf0be392))

Separate tuner-state recovery from final-model checkpoint resume so interrupted jobs can restart
  without losing completed trials, and switch the SLURM wrappers to stable long-form flags that
  point back to the same saved state.

### Chores

- Add guidelines for brief Conventional Commits and SLURM-safe execution
  ([`d224c71`](https://github.com/joshbruegger/GrainSegmentation/commit/d224c7198cbde8f9b045e967ed82af8815ebe29b))

- Introduced a new markdown file outlining the use of brief Conventional Commits for commit
  messages, emphasizing clarity and specificity. - Added a markdown file detailing SLURM-safe
  execution practices, including command usage and GPU allocation recommendations. - Updated
  existing documentation to reflect the preferred use of `uv` for Python and package management.

### Features

- Filter TensorFlow stderr in SLURM jobs, add --verbose and training summaries
  ([`29f5677`](https://github.com/joshbruegger/GrainSegmentation/commit/29f56770f952fafc3f4d2681e91c32933684a34b))


## v0.7.0 (2026-03-09)

### Chores

- Remove outdated project reorganization plans
  ([`ce78630`](https://github.com/joshbruegger/GrainSegmentation/commit/ce7863073bb163e45e18375004b9e343aa4a701d))

- Deleted the project reorganization design and implementation plan documents as they are no longer
  relevant to the current project structure.

- Update Python version and enhance README documentation
  ([`3fbe2e6`](https://github.com/joshbruegger/GrainSegmentation/commit/3fbe2e6fcd0b0dcf6d3e34d6b668bf26b1cdce8a))

- Changed Python version from 3.12.3 to 3.12 in the .python-version file. - Updated README to
  clarify the objective and details of the grain segmentation models, including the addition of the
  PPL+PPX composite. - Added a new markdown file for using `uv` as the default Python package
  manager and execution tool. - Modified SLURM script to increase memory allocation and job time for
  PPL + All PPX training jobs.

### Features

- Enhance data validation and error handling in dataset building
  ([`f117b41`](https://github.com/joshbruegger/GrainSegmentation/commit/f117b416e2a20e7e3defd4e9b90551aeccaaca12))

- Introduced validation functions to ensure input images and masks meet required conditions,
  including shape consistency and valid class IDs. - Added error handling for non-positive patch
  size and stride values. - Updated dataset building logic to incorporate new validation checks,
  improving robustness and user feedback during data preparation.

- Implement evaluation framework for grain segmentation models
  ([`435cee2`](https://github.com/joshbruegger/GrainSegmentation/commit/435cee244e73eee425650557bbbdf85ba0375936))

- Added a comprehensive evaluation module including metrics computation (IoU, F1, AJI) and model
  predictions. - Introduced a script for evaluating trained models on test images, with options for
  saving predictions and metrics. - Developed a plotting utility for generating quantitative plots
  and qualitative overlays of model performance. - Enhanced README documentation to detail
  evaluation metrics and their significance in the context of grain boundary segmentation. -
  Included unit tests to ensure the correctness of evaluation functions and metrics calculations.

### Refactoring

- Simplify U-Net model configuration and training parameters
  ([`725c423`](https://github.com/joshbruegger/GrainSegmentation/commit/725c423ccf8b77eaf18466591506ce25f3e0d560))

- Set default base_filters to 16 in the U-Net model to streamline configuration. - Updated batch
  size handling in the CVTuner class to use a predefined best_batch_size instead of a hyperparameter
  choice. - Refactored training logic to improve clarity and maintainability, including adjustments
  to the dataset builder function and model compilation process.

- Update SLURM scripts and training parameters
  ([`d263aa3`](https://github.com/joshbruegger/GrainSegmentation/commit/d263aa3a6aa19ae74a707945b229a4850bf1b5c5))

- Removed CPU allocation settings from SLURM submission scripts for PPL jobs. - Increased job time
  limit in the training script from 5 to 12 hours. - Adjusted default folds for cross-validation
  from 5 to 2 in the training script. - Reduced tuning epochs from 30 to 20 to streamline training
  process. - Updated maximum hyperparameter tuning trials from 10 to 7 for better efficiency.


## v0.6.0 (2026-03-06)

### Features

- Update README and SLURM scripts for training configuration
  ([`27fea94`](https://github.com/joshbruegger/GrainSegmentation/commit/27fea940f0ec74273a0277786775f90619f6bfcd))

- Updated README to include dropout rate in the Bayesian Optimization Loop description. - Enhanced
  SLURM submission scripts by adding a `-t` flag to skip tuning during job submission. - Adjusted
  resource allocation in SLURM scripts for improved performance and modified job time settings.


## v0.5.0 (2026-03-06)

### Features

- Enhance U-Net model with dropout regularization
  ([`da9f4b4`](https://github.com/joshbruegger/GrainSegmentation/commit/da9f4b462251613ded59bb42ca86c9c6d96b1646))

- Added Dropout layer support in the U-Net architecture to improve generalization. - Updated
  base_filters option to include 64 for more flexibility in model configuration. - Introduced a new
  hyperparameter for dropout rate, allowing dynamic adjustment during training.


## v0.4.0 (2026-03-06)

### Features

- Add log plotting script for training metrics
  ([`efd027d`](https://github.com/joshbruegger/GrainSegmentation/commit/efd027d4908755d21c18e467d2760ab2948fffee))

- Introduced `plot_training_log.py` to visualize training and validation accuracy and loss over
  epochs. - Implemented log parsing using regex to extract relevant metrics from Keras training
  logs. - Added functionality to save generated plots as PNG files for easy analysis.

- Enhance SLURM scripts and training configuration
  ([`d53b169`](https://github.com/joshbruegger/GrainSegmentation/commit/d53b169d1a8941e9d0772b33b5717aec568a8616))

- Added a new flag `-c` to the SLURM submission script for resuming training from the latest model
  checkpoint. - Increased GPU allocation in the training script from 1 to 2 for improved
  performance. - Adjusted the tuning epochs from 20 to 30 to enhance model training. - Updated the
  training logic to dynamically determine the number of GPUs available and adjust batch sizes
  accordingly. - Added new dependencies for `tensorboard` and `setuptools` in the project
  configuration.


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
