import datetime
import math
import os
import statistics
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
import warnings
from data import build_dataset, list_samples, create_spatial_cv_folds
from model import build_unet, initialize_from_checkpoint, weighted_crossentropy

warnings.filterwarnings("ignore", message="Your input ran out of data")


def print_section(title: str) -> None:
    border = "=" * 80
    print(f"\n{border}")
    print(title)
    print(border)


def print_key_values(items) -> None:
    entries = list(items.items()) if hasattr(items, "items") else list(items)
    if not entries:
        return

    key_width = max(len(str(key)) for key, _ in entries)
    for key, value in entries:
        print(f"{key:<{key_width}} : {value}")


def summarize_training_sources(samples):
    grouped_sources = []
    source_index = {}

    for sample in samples:
        sample_id = sample.get("id", f"sample_{len(grouped_sources) + 1}")
        image_paths = tuple(sample.get("images", ()))
        source_key = (sample_id, image_paths)

        if source_key not in source_index:
            source_index[source_key] = len(grouped_sources)
            grouped_sources.append(
                {
                    "id": sample_id,
                    "images": list(image_paths),
                    "region_count": 0,
                }
            )

        if "region" in sample:
            grouped_sources[source_index[source_key]]["region_count"] += 1

    return grouped_sources


def print_training_image_paths(samples, title: str) -> None:
    grouped_sources = summarize_training_sources(samples)
    print_section(title)
    print_key_values(
        [
            ("total_entries", len(samples)),
            ("unique_source_samples", len(grouped_sources)),
        ]
    )
    for sample_index, sample in enumerate(grouped_sources, start=1):
        print(f"Sample {sample_index}: {sample['id']}")
        if sample["region_count"] > 0:
            print(f"  region_count : {sample['region_count']}")
        for input_index, image_path in enumerate(sample["images"], start=1):
            print(f"  input[{input_index}] : {image_path}")


def has_saved_tuner_state(tuning_dir: str, run_name: str, num_inputs: int) -> bool:
    tuner_state_dir = os.path.join(
        tuning_dir,
        f"tuning_{run_name}_{num_inputs}in",
        "unet_tuning",
    )
    return os.path.exists(os.path.join(tuner_state_dir, "oracle.json"))


def find_optimal_batch_size(model_builder, train_dataset_fn, start_batch_size=32):
    batch_size = start_batch_size
    while batch_size >= 1:
        print(f"Testing batch size: {batch_size}")
        try:
            model = model_builder()
            dataset = train_dataset_fn(batch_size)
            model.fit(dataset, steps_per_epoch=1, epochs=1, verbose=0)
            print(f"Successfully ran with batch size: {batch_size}")
            return batch_size
        except tf.errors.ResourceExhaustedError:
            print(f"OOM with batch size {batch_size}. Halving batch size.")
            batch_size //= 2

    raise RuntimeError(
        "Could not find a batch size that fits in memory, even batch_size=1 failed."
    )


class CVTuner(kt.BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_trial(
        self,
        trial,
        cv_folds,
        patch_size,
        stride,
        epochs,
        num_inputs,
        best_batch_size,
        **kwargs,
    ):
        hp_batch_size = best_batch_size

        val_losses = []
        for i, (train_samples, val_samples) in enumerate(cv_folds):
            strategy = tf.distribute.MirroredStrategy()
            global_batch_size = hp_batch_size * strategy.num_replicas_in_sync
            # Create a unique cache file prefix for each fold and trial
            tmpdir = os.environ.get("TMPDIR", "/tmp")
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            train_cache = os.path.join(
                tmpdir, f"tf_cache_train_fold{i}_trial{trial.trial_id}_{job_id}"
            )
            val_cache = os.path.join(
                tmpdir, f"tf_cache_val_fold{i}_trial{trial.trial_id}_{job_id}"
            )

            train_dataset = build_dataset(
                train_samples,
                patch_size=patch_size,
                stride=stride,
                batch_size=global_batch_size,
                augment=True,
                num_inputs=num_inputs,
                cache_file=train_cache,
            )
            val_dataset = build_dataset(
                val_samples,
                patch_size=patch_size,
                stride=patch_size,  # Validation stride = patch_size for 0% overlap
                batch_size=global_batch_size,
                augment=False,
                num_inputs=num_inputs,
                cache_file=val_cache,
            )

            with strategy.scope():
                model = self.hypermodel.build(trial.hyperparameters)
                learning_rate = trial.hyperparameters.Float(
                    "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
                )

                if i == 0:
                    print(f"\n--- Details for Trial {trial.trial_id} ---")
                    for hp_name, hp_value in trial.hyperparameters.values.items():
                        print(f"{hp_name}: {hp_value}")
                    print("-----------------------------------\n")

                optimizer = Adam(learning_rate=learning_rate)

                model.compile(
                    optimizer=optimizer,
                    loss=weighted_crossentropy,
                    metrics=["accuracy"],
                )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )

            try:
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=[early_stopping],
                    verbose=2,
                )
                val_losses.append(min(history.history["val_loss"]))
            except tf.errors.ResourceExhaustedError:
                print(f"OOM with batch size {hp_batch_size}, marking trial as failure.")
                return {"val_loss": 1e9}

        return {"val_loss": np.mean(val_losses)}


def load_saved_model(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"weighted_crossentropy": weighted_crossentropy},
    )


def build_model_for_tuning(
    checkpoint_path: str | None,
    resume_path: str | None,
    patch_size: int,
    num_inputs: int,
    hp=None,
):
    del resume_path
    if checkpoint_path:
        return initialize_from_checkpoint(
            checkpoint_path, patch_size, num_inputs=num_inputs, hp=hp
        )
    return build_unet(patch_size, num_inputs=num_inputs, hp=hp)


def build_final_model(
    checkpoint_path: str | None,
    resume_path: str | None,
    patch_size: int,
    num_inputs: int,
    hp=None,
):
    if resume_path:
        return load_saved_model(resume_path)
    return build_model_for_tuning(
        checkpoint_path=checkpoint_path,
        resume_path=None,
        patch_size=patch_size,
        num_inputs=num_inputs,
        hp=hp,
    )


def compile_model_for_training(model, learning_rate: float):
    if getattr(model, "optimizer", None) is not None:
        print("Resuming from checkpoint, retaining optimizer state.")
        return model

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=weighted_crossentropy,
        metrics=["accuracy"],
    )
    return model


def infer_initial_epoch(model, dataset) -> int:
    if getattr(model, "optimizer", None) is None:
        return 0

    iterations = model.optimizer.iterations.numpy()
    if iterations <= 0:
        return 0

    print("Calculating dataset length for resuming...")
    steps_per_epoch = sum(1 for _ in dataset)
    if steps_per_epoch <= 0:
        return 0

    initial_epoch = int(iterations // steps_per_epoch)
    print(f"Resuming training from epoch {initial_epoch}")
    return initial_epoch


def select_best_epoch_from_history(history, monitor: str = "val_loss"):
    metric_values = list(getattr(history, "history", {}).get(monitor, []))
    if not metric_values:
        raise ValueError(f"History does not contain any values for {monitor}.")

    best_index = min(range(len(metric_values)), key=metric_values.__getitem__)
    return best_index + 1, metric_values[best_index]


def summarize_cv_epoch_selection(
    fold_summaries,
    full_steps_per_epoch: int,
    epoch_cap: int | None,
):
    if not fold_summaries:
        raise ValueError("fold_summaries must not be empty.")
    if full_steps_per_epoch <= 0:
        raise ValueError("full_steps_per_epoch must be > 0.")
    if epoch_cap is not None and epoch_cap <= 0:
        raise ValueError("epoch_cap must be > 0 when provided.")

    summarized_folds = []
    fold_best_updates = []
    for fold_summary in fold_summaries:
        best_epoch = int(fold_summary["best_epoch"])
        train_steps_per_epoch = int(fold_summary["train_steps_per_epoch"])
        if best_epoch <= 0:
            raise ValueError("best_epoch must be > 0 for every fold.")
        if train_steps_per_epoch <= 0:
            raise ValueError("train_steps_per_epoch must be > 0 for every fold.")

        best_updates = best_epoch * train_steps_per_epoch
        fold_best_updates.append(best_updates)
        summarized_folds.append({**fold_summary, "best_updates": best_updates})

    aggregated_updates = math.ceil(statistics.median(fold_best_updates))
    unbounded_final_epochs = max(
        1, math.ceil(aggregated_updates / full_steps_per_epoch)
    )
    selected_final_epochs = unbounded_final_epochs
    if epoch_cap is not None:
        selected_final_epochs = min(selected_final_epochs, epoch_cap)

    return {
        "fold_summaries": summarized_folds,
        "aggregated_updates": aggregated_updates,
        "unbounded_final_epochs": unbounded_final_epochs,
        "selected_final_epochs": max(1, int(selected_final_epochs)),
        "epoch_cap": epoch_cap,
        "used_fallback": False,
    }


def build_epoch_selection_fallback(
    epoch_cap: int,
    reason: str,
    *,
    fold_summaries=None,
):
    return {
        "fold_summaries": list(fold_summaries or []),
        "aggregated_updates": None,
        "unbounded_final_epochs": None,
        "selected_final_epochs": max(1, int(epoch_cap)),
        "epoch_cap": epoch_cap,
        "used_fallback": True,
        "fallback_reason": reason,
    }


def estimate_final_epochs_from_cv(
    *,
    cv_folds,
    checkpoint_path: str | None,
    patch_size: int,
    stride: int,
    num_inputs: int,
    best_batch_size: int,
    best_hp,
    max_epochs: int,
    full_steps_per_epoch: int,
):
    if full_steps_per_epoch <= 0:
        raise ValueError("full_steps_per_epoch must be > 0.")

    fold_summaries = []
    learning_rate = best_hp.get("learning_rate")
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    for fold_index, (train_samples, val_samples) in enumerate(cv_folds, start=1):
        strategy = tf.distribute.MirroredStrategy()
        global_batch_size = best_batch_size * strategy.num_replicas_in_sync
        train_cache = os.path.join(
            tmpdir, f"tf_cache_epoch_select_train_fold{fold_index}_{job_id}"
        )
        val_cache = os.path.join(
            tmpdir, f"tf_cache_epoch_select_val_fold{fold_index}_{job_id}"
        )

        train_dataset = build_dataset(
            train_samples,
            patch_size=patch_size,
            stride=stride,
            batch_size=global_batch_size,
            augment=True,
            num_inputs=num_inputs,
            cache_file=train_cache,
        )
        val_dataset = build_dataset(
            val_samples,
            patch_size=patch_size,
            stride=patch_size,
            batch_size=global_batch_size,
            augment=False,
            num_inputs=num_inputs,
            cache_file=val_cache,
        )
        train_steps_per_epoch = sum(1 for _ in train_dataset)

        with strategy.scope():
            model = build_model_for_tuning(
                checkpoint_path=checkpoint_path,
                resume_path=None,
                patch_size=patch_size,
                num_inputs=num_inputs,
                hp=best_hp,
            )
            model = compile_model_for_training(model, learning_rate=learning_rate)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        )

        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=max_epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=2,
            )
        except tf.errors.ResourceExhaustedError:
            return build_epoch_selection_fallback(
                max_epochs,
                f"OOM during frozen CV epoch selection on fold {fold_index}.",
                fold_summaries=fold_summaries,
            )

        best_epoch, best_val_loss = select_best_epoch_from_history(history)
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_steps_per_epoch": train_steps_per_epoch,
            }
        )

    return summarize_cv_epoch_selection(
        fold_summaries=fold_summaries,
        full_steps_per_epoch=full_steps_per_epoch,
        epoch_cap=max_epochs,
    )


def create_tuner(
    hypermodel,
    max_trials: int,
    tuning_dir: str,
    run_name: str,
    num_inputs: int,
):
    return CVTuner(
        hypermodel=hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        directory=os.path.join(tuning_dir, f"tuning_{run_name}_{num_inputs}in"),
        project_name="unet_tuning",
        overwrite=False,
    )


def train_model(
    image_dir: str,
    mask_dir: str,
    checkpoint_path: str | None,
    resume_path: str | None,
    output_model_path: str,
    patch_size: int,
    stride: int,
    tune_epochs: int,
    final_epochs: int,
    image_suffixes: list[str],
    mask_ext: str | None,
    mask_stem_suffix: str,
    split_tile_size: int,
    split_coverage_bins: int,
    num_inputs: int,
    run_name: str,
    tuning_dir: str,
    n_splits: int,
    random_state: int,
    use_mixed_precision: bool,
    max_trials: int,
    skip_tuning: bool = False,
) -> tf.keras.Model:
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    num_gpus = len(tf.config.list_logical_devices("GPU"))
    print_section("Preparing training pipeline")
    print_key_values(
        [
            ("num_gpus", num_gpus),
            ("num_inputs", num_inputs),
            ("patch_size", patch_size),
            ("stride", stride),
            ("run_name", run_name),
        ]
    )
    num_replicas_in_sync = num_gpus if num_gpus > 0 else 1

    samples = list_samples(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_suffixes=image_suffixes,
        mask_ext=mask_ext,
        mask_stem_suffix=mask_stem_suffix,
        num_inputs=num_inputs,
    )

    cv_folds = create_spatial_cv_folds(
        samples,
        tile_size=split_tile_size,
        n_splits=n_splits,
        random_state=random_state,
        coverage_bins=split_coverage_bins,
    )

    def build_hypermodel(hp=None):
        return build_model_for_tuning(
            checkpoint_path=checkpoint_path,
            resume_path=resume_path,
            patch_size=patch_size,
            num_inputs=num_inputs,
            hp=hp,
        )

    def compiled_model_builder():
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = (
                build_final_model(
                    checkpoint_path=checkpoint_path,
                    resume_path=resume_path,
                    patch_size=patch_size,
                    num_inputs=num_inputs,
                    hp=None,
                )
                if skip_tuning
                else build_hypermodel(hp=None)
            )
            return compile_model_for_training(model, learning_rate=1e-3)

    train_samples_fold0 = cv_folds[0][0]
    print_training_image_paths(
        train_samples_fold0,
        "Training image order for batch-size probe",
    )

    def dataset_builder_fn(bs):
        global_bs = bs * num_replicas_in_sync
        return build_dataset(
            train_samples_fold0,
            patch_size=patch_size,
            stride=stride,
            batch_size=global_bs,
            augment=True,
            num_inputs=num_inputs,
            cache_file=None,  # Don't write to disk for just 1 step tests
        )

    print_section("Testing batch size")
    best_batch_size = find_optimal_batch_size(
        compiled_model_builder, dataset_builder_fn, start_batch_size=32
    )
    print(f"Optimal per-replica batch size found: {best_batch_size}")
    global_batch_size = best_batch_size * num_replicas_in_sync
    selected_hyperparameters = []
    epoch_selection_summary = None

    if skip_tuning:
        print("Skipping hyperparameter tuning. Using default settings.")
        train_samples, val_samples = cv_folds[0]
        learning_rate = 1e-3
        selected_hyperparameters = [("learning_rate", learning_rate)]
        print_training_image_paths(
            train_samples,
            "Training image order for final training",
        )

        tmpdir = os.environ.get("TMPDIR", "/tmp")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        full_dataset = build_dataset(
            train_samples,
            patch_size=patch_size,
            stride=stride,
            batch_size=global_batch_size,
            augment=True,
            num_inputs=num_inputs,
            cache_file=os.path.join(tmpdir, f"tf_cache_train_full_{job_id}"),
        )

        val_dataset = build_dataset(
            val_samples,
            patch_size=patch_size,
            stride=patch_size,
            batch_size=global_batch_size,
            augment=False,
            num_inputs=num_inputs,
            cache_file=os.path.join(tmpdir, f"tf_cache_val_full_{job_id}"),
        )

        final_model = compiled_model_builder()
        monitor_metric = "val_loss"
        fit_kwargs = {"validation_data": val_dataset}

    else:
        tuner = create_tuner(
            hypermodel=build_hypermodel,
            max_trials=max_trials,
            tuning_dir=tuning_dir,
            run_name=run_name,
            num_inputs=num_inputs,
        )
        tuning_title = (
            "Resuming tuning"
            if has_saved_tuner_state(tuning_dir, run_name, num_inputs)
            else "Starting tuning"
        )
        for fold_index, (fold_train_samples, _) in enumerate(cv_folds, start=1):
            print_training_image_paths(
                fold_train_samples,
                f"Tuning fold {fold_index} training image order",
            )
        print_section(tuning_title)
        print_key_values(
            [
                ("tuning_dir", tuning_dir),
                ("run_name", run_name),
                ("max_trials", max_trials),
                ("tune_epochs", tune_epochs),
                ("per_replica_batch_size", best_batch_size),
                ("global_batch_size", global_batch_size),
            ]
        )

        tuner.search(
            cv_folds=cv_folds,
            patch_size=patch_size,
            stride=stride,
            epochs=tune_epochs,
            num_inputs=num_inputs,
            best_batch_size=best_batch_size,
        )

        best_hp = tuner.get_best_hyperparameters()[0]
        print(f"Best hyperparameters: {best_hp.values}")

        learning_rate = best_hp.get("learning_rate")
        selected_hyperparameters = list(best_hp.values.items())

        all_train_samples = []
        for _, val_samples_fold in cv_folds:
            all_train_samples.extend(val_samples_fold)
        print_training_image_paths(
            all_train_samples,
            "Training image order for final training",
        )

        tmpdir = os.environ.get("TMPDIR", "/tmp")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        full_dataset = build_dataset(
            all_train_samples,
            patch_size=patch_size,
            stride=stride,
            batch_size=global_batch_size,
            augment=True,
            num_inputs=num_inputs,
            cache_file=os.path.join(tmpdir, f"tf_cache_train_all_folds_{job_id}"),
        )
        full_steps_per_epoch = sum(1 for _ in full_dataset)
        if resume_path and not checkpoint_path:
            epoch_selection_summary = build_epoch_selection_fallback(
                final_epochs,
                "Frozen CV epoch selection skipped for resumed tuned training "
                "without a checkpoint baseline.",
            )
        else:
            print_section("Starting frozen CV epoch selection")
            try:
                epoch_selection_summary = estimate_final_epochs_from_cv(
                    cv_folds=cv_folds,
                    checkpoint_path=checkpoint_path,
                    patch_size=patch_size,
                    stride=stride,
                    num_inputs=num_inputs,
                    best_batch_size=best_batch_size,
                    best_hp=best_hp,
                    max_epochs=final_epochs,
                    full_steps_per_epoch=full_steps_per_epoch,
                )
            except ValueError as exc:
                epoch_selection_summary = build_epoch_selection_fallback(
                    final_epochs,
                    f"Frozen CV epoch selection skipped: {exc}",
                )
        final_epochs = epoch_selection_summary["selected_final_epochs"]

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            final_model = build_final_model(
                checkpoint_path=checkpoint_path,
                resume_path=resume_path,
                patch_size=patch_size,
                num_inputs=num_inputs,
                hp=best_hp,
            )
            final_model = compile_model_for_training(final_model, learning_rate)

        monitor_metric = "loss"
        fit_kwargs = {}

    log_dir = os.path.join(
        "logs", "fit", f"{run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch"
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_model_path.replace(".keras", "_best.keras"),
        monitor=monitor_metric,
        save_best_only=True,
        verbose=1,
    )
    checkpoint_latest = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_model_path.replace(".keras", "_latest.keras"),
        save_best_only=False,
        verbose=1,
    )

    initial_epoch = 0
    should_run_final_training = True
    if resume_path:
        print_section("Resuming training")
        print_key_values([("resume_path", resume_path)])
        initial_epoch = infer_initial_epoch(final_model, full_dataset)
        if initial_epoch >= final_epochs:
            print(
                "Resume checkpoint already meets or exceeds the selected final "
                "epoch budget. Skipping additional training."
            )
            should_run_final_training = False

    if epoch_selection_summary is not None:
        print_section("Frozen CV epoch selection")
        epoch_selection_details = [
            ("epoch_cap", epoch_selection_summary.get("epoch_cap")),
            ("aggregated_updates", epoch_selection_summary.get("aggregated_updates")),
            (
                "unbounded_final_epochs",
                epoch_selection_summary.get("unbounded_final_epochs"),
            ),
            (
                "selected_final_epochs",
                epoch_selection_summary["selected_final_epochs"],
            ),
            ("used_fallback", epoch_selection_summary["used_fallback"]),
        ]
        if epoch_selection_summary["used_fallback"]:
            epoch_selection_details.append(
                ("fallback_reason", epoch_selection_summary["fallback_reason"])
            )
        print_key_values(epoch_selection_details)
        for fold_summary in epoch_selection_summary["fold_summaries"]:
            print_key_values(
                [
                    ("fold_index", fold_summary["fold_index"]),
                    ("best_epoch", fold_summary["best_epoch"]),
                    ("best_val_loss", fold_summary.get("best_val_loss")),
                    ("train_steps_per_epoch", fold_summary["train_steps_per_epoch"]),
                    ("best_updates", fold_summary.get("best_updates")),
                ]
            )

    final_training_details = list(selected_hyperparameters)
    final_training_details.extend(
        [
            ("per_replica_batch_size", best_batch_size),
            ("global_batch_size", global_batch_size),
            ("final_epochs", final_epochs),
            ("monitor_metric", monitor_metric),
            ("output_model_path", output_model_path),
        ]
    )
    print_section("Starting final training")
    print_key_values(final_training_details)

    if should_run_final_training:
        final_model.fit(
            full_dataset,
            epochs=final_epochs,
            initial_epoch=initial_epoch,
            callbacks=[
                reduce_lr,
                checkpoint_best,
                checkpoint_latest,
                tensorboard_callback,
            ],
            verbose=2,
            **fit_kwargs,
        )

    final_model.save(output_model_path)
    return final_model
