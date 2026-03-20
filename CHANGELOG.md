# CHANGELOG


## v0.20.0 (2026-03-20)

### Features

- **evaluation**: Instance masks from semantic preds (CC + watershed)
  ([`3e7a81a`](https://github.com/joshbruegger/GrainSegmentation/commit/3e7a81a1e8b724b28970c38acafbf412c4d2d7b0))

### Refactoring

- **yolo**: Simplify eval to val/sahi, rename sahi-coco
  ([`676bb09`](https://github.com/joshbruegger/GrainSegmentation/commit/676bb099f9d711c38d3b8493fa237eb3f077fde3))


## v0.19.0 (2026-03-19)

### Bug Fixes

- Fixed outdated tests
  ([`4d012bd`](https://github.com/joshbruegger/GrainSegmentation/commit/4d012bdab0b1e9972bd8ebd3c33b12cdd1a2779c))

### Features

- Added yolo evaluation
  ([`afd0236`](https://github.com/joshbruegger/GrainSegmentation/commit/afd02367983908d0704cadc4a9516a681b610799))


## v0.18.0 (2026-03-19)

### Bug Fixes

- Add missing dep declaration in evaluation package
  ([`f97035a`](https://github.com/joshbruegger/GrainSegmentation/commit/f97035aaa40f5fa096606d3867a4ff061e6708ac))

- Change training time
  ([`787134b`](https://github.com/joshbruegger/GrainSegmentation/commit/787134b787d5fb2dfe82c082dc850fdd398878e0))

### Features

- Add script to make tuning figures
  ([`2f2cd33`](https://github.com/joshbruegger/GrainSegmentation/commit/2f2cd337eb6e17b28ad29bfe03f3d8ccf619c307))


## v0.17.0 (2026-03-18)

### Features

- Enhance YOLO training and tuning capabilities
  ([`70ab526`](https://github.com/joshbruegger/GrainSegmentation/commit/70ab52658c12a6dc47da49e49c65f538bd4477d8))

- Added hyperparameter tuning options in `train_yolo26x_seg.sh` and `train.py`, allowing users to
  specify tuning epochs and iterations. - Updated `pipeline.py` to include a new `tune_model`
  function for executing Ultralytics' built-in hyperparameter tuning with a custom search space. -
  Modified `submit_yolo_experiments.sh` to support a new `--tune` argument for submitting tuning
  jobs. - Adjusted README documentation to reflect new tuning features and usage instructions. -
  Updated tests to cover new tuning functionality and ensure correct behavior.

- Enhance YOLO training scripts with learning rate and dropout parameters
  ([`b5acae5`](https://github.com/joshbruegger/GrainSegmentation/commit/b5acae5d3a749070b650fa4d4866192c449d0862))

- Updated `submit_yolo_experiments.sh` to accept learning rate and dropout as arguments for job
  submissions. - Modified `train_yolo26x_seg.sh` to include command-line options for learning rate
  and dropout. - Adjusted `pipeline.py` and `train.py` to utilize the new learning rate and dropout
  parameters during model training. - Updated unit tests in `test_pipeline.py` to validate the new
  parameters in training and tuning functions.


## v0.16.0 (2026-03-15)

### Bug Fixes

- Adjusted batch size and epochs
  ([`ef29669`](https://github.com/joshbruegger/GrainSegmentation/commit/ef296690702c0407bcfbd6895863016a5bc60336))

- Update dataset YAML path handling in YOLO training script
  ([`9cc1616`](https://github.com/joshbruegger/GrainSegmentation/commit/9cc161602a343c23e08815a4bffa1942181e40f4))

- Modified `train_yolo26x_seg.sh` to copy the dataset YAML to a temporary directory and update its
  path to ensure local resolution of images. - Added a Python script to rewrite the copied YAML file
  with the correct dataset path. - Updated file permissions for `submit_yolo_experiments.sh` to make
  it executable. - Introduced a new test case in `test_slurm_scripts.py` to verify the YAML path
  rewriting functionality.

### Features

- Add script to split stacked TIFF channels into RGB TIFFs
  ([`859d006`](https://github.com/joshbruegger/GrainSegmentation/commit/859d006846d37875ad871cbd81075335b0bc95a6))

- Introduced `split_tiff_channels.py` to split a stacked TIFF file into individual RGB TIFF files
  based on channel triplets. - Implemented command-line interface for specifying input file, output
  directory, and optional filename prefix. - Added validation for input file format and channel
  count. - Created unit tests in `test_split_tiff_channels.py` to ensure correct functionality and
  error handling.


## v0.15.0 (2026-03-15)

### Features

- Add visualization script for YOLO dataset
  ([`dada6b5`](https://github.com/joshbruegger/GrainSegmentation/commit/dada6b515bf3dda8e3474a21b0351f963ba232a7))

- Introduced `visualize_dataset.py` to generate visualizations of annotated YOLO segmentation
  samples. - Implemented argument parsing for dataset directory, output directory, number of
  samples, and random seed. - Added functions to load dataset configuration, collect samples, and
  save visualizations with annotations. - Created unit tests in `test_visualize_dataset.py` to
  validate functionality and ensure correct behavior.

- Add yolo26 training pipeline
  ([`8e9a6c1`](https://github.com/joshbruegger/GrainSegmentation/commit/8e9a6c11b8dca249e86c38d8fb52601cbf6fe4c1))


## v0.14.0 (2026-03-14)

### Features

- Add patchify script for splitting TIFF and GPKG files into YOLO format
  ([`58c2abf`](https://github.com/joshbruegger/GrainSegmentation/commit/58c2abfc63ab39787c1e1918eb6ee1dc40239323))

- Introduced `patchify.sh` to automate the process of copying input files, running the split script
  for multiple models, and organizing output into the specified directory structure. - Updated
  `train_unet_multi_input.sh` to include a new argument `--split-tile-size` for specifying the tile
  size during training. - Modified argument parsing in `train_unet_multi_input.py` to accommodate
  the new `--split-tile-size` option and adjusted default values for `--patch-size` and
  `--split-tile-size` for consistency.


## v0.13.0 (2026-03-14)

### Features

- Add raster-to-polygon slurm job
  ([`29dd8e1`](https://github.com/joshbruegger/GrainSegmentation/commit/29dd8e1c9647c72c93c629f94c1a2d416791ff1c))

- Add tiff channel stacking script
  ([`dc4abed`](https://github.com/joshbruegger/GrainSegmentation/commit/dc4abedcd334271a0776a9b08a678efb1273b744))


## v0.12.0 (2026-03-14)

### Features

- Add raster-to-gpkg export
  ([`4c997ec`](https://github.com/joshbruegger/GrainSegmentation/commit/4c997ecf1f0f5f9b379c284795465621b1ca3c2a))

- Add split TIFF+GPKG to YOLO train/val export script
  ([`79f9330`](https://github.com/joshbruegger/GrainSegmentation/commit/79f9330fc1b349b37268c071ea2a130770592457))

- Enhance SLURM scripts for raster processing and add new utilities
  ([`539eb3d`](https://github.com/joshbruegger/GrainSegmentation/commit/539eb3d1d7aa7c6d0995e3af004f46e5fb6ac853))

- Updated `evaluate_models_and_plot.sh` to include default mask extension and stem suffix. - Added
  `merge_multichannel_tiff.sh` for merging TIFF images into a multichannel format. - Introduced
  `raster_to_polygon.sh` for converting raster images to polygons with enhanced error handling and
  directory processing. - Implemented `stack_tiff_channels.py` for stacking TIFF channels and added
  corresponding tests. - Updated `.gitignore` to exclude new directories and files.


## v0.11.0 (2026-03-11)

### Bug Fixes

- Split qualitative overlays by model
  ([`600c5ac`](https://github.com/joshbruegger/GrainSegmentation/commit/600c5ac6cb212a5dc99cc4c6bd919f8a254cbca1))

Make overlay outputs easier to inspect on large TIFF evaluations by writing one red-tinted image per
  model and ground truth instead of a single montage.

### Features

- Add reusable evaluation slurm workflow
  ([`5565f3c`](https://github.com/joshbruegger/GrainSegmentation/commit/5565f3ca4886deb273d92673e0079320f7ad19d7))

### Refactoring

- Use spatial holdout validation
  ([`863cca8`](https://github.com/joshbruegger/GrainSegmentation/commit/863cca8120aec7c6c3217bef8066e8cc921451c6))

Align tuning and final training around a single spatial validation holdout so early stopping and
  checkpointing use the same signal. Keep low-coverage edge regions out of validation to make the
  holdout more informative.


## v0.10.0 (2026-03-10)

### Bug Fixes

- Update TensorFlow stderr filter patterns and adjust SLURM job memory allocation
  ([`fdd175b`](https://github.com/joshbruegger/GrainSegmentation/commit/fdd175b759776b6e6b373b84b4770394e027a2a8))

- Modified the regex pattern for unrecognized features in TensorFlow stderr to improve clarity. -
  Changed memory allocation parameters in SLURM job submissions to optimize resource usage.

### Chores

- Log frozen CV epoch selection start
  ([`8c3faa2`](https://github.com/joshbruegger/GrainSegmentation/commit/8c3faa21a6491c6f3de8c02e090b3489b7e3147d))

### Documentation

- Update README to clarify final test evaluation and ablation study interpretation
  ([`11fbbf1`](https://github.com/joshbruegger/GrainSegmentation/commit/11fbbf1d525a9bce993006872d353fde3874a604))

- Revised the description of the final test to specify that it evaluates a single held-out section,
  emphasizing the descriptive nature of the results. - Added clarification on the interpretation of
  ablation study comparisons, highlighting their case-study style rather than statistical evidence.
  - Updated results and discussion sections to reflect the limitations of using a single test image
  for formal statistical comparisons.

### Features

- Reuse fold caches during tuning
  ([`390b542`](https://github.com/joshbruegger/GrainSegmentation/commit/390b5426135e929abed705355b770f82769a8093))

### Refactoring

- Align evaluation reporting with single-image test
  ([`ea99a5c`](https://github.com/joshbruegger/GrainSegmentation/commit/ea99a5c6ef8fb98f1a0682099afd89964ba19ec8))

Make one-sample evaluation outputs descriptive so the JSON and plots no longer imply aggregate or
  inferential comparisons that the test setup cannot support.


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
