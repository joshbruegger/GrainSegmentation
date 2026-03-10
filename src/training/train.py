import datetime
import hashlib
import json
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
import warnings
from data import build_dataset, list_samples, create_spatial_holdout_split
from model import build_unet, initialize_from_checkpoint, weighted_crossentropy

warnings.filterwarnings("ignore", message="Your input ran out of data")

FOLD_CACHE_SCHEMA_VERSION = 1


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


def _validation_fraction_tag(validation_fraction: float) -> str:
    return f"val{int(round(validation_fraction * 100))}"


def _tuner_state_dir(
    tuning_dir: str,
    run_name: str,
    num_inputs: int,
    validation_fraction: float,
) -> str:
    return os.path.join(
        tuning_dir,
        f"tuning_{run_name}_{num_inputs}in_{_validation_fraction_tag(validation_fraction)}",
        "unet_tuning",
    )


def has_saved_tuner_state(
    tuning_dir: str,
    run_name: str,
    num_inputs: int,
    validation_fraction: float,
) -> bool:
    tuner_state_dir = _tuner_state_dir(
        tuning_dir,
        run_name,
        num_inputs,
        validation_fraction,
    )
    return os.path.exists(os.path.join(tuner_state_dir, "oracle.json"))


def _fold_cache_scope() -> tuple[str, str]:
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return tmpdir, job_id
    return tmpdir, f"local-{os.getpid()}"


def _fold_cache_sample_descriptor(samples):
    descriptors = []
    for sample in samples:
        if not isinstance(sample, dict):
            descriptors.append({"value": repr(sample)})
            continue

        descriptor = {
            "id": sample.get("id"),
            "images": list(sample.get("images", ())),
            "mask": sample.get("mask"),
        }
        if "region" in sample:
            descriptor["region"] = list(sample["region"])
        descriptors.append(descriptor)
    return descriptors


def build_fold_cache_path(
    samples,
    *,
    role: str,
    patch_size: int,
    stride: int,
    num_inputs: int,
) -> str:
    if role not in {"train", "val"}:
        raise ValueError("role must be 'train' or 'val'")

    tmpdir, cache_scope = _fold_cache_scope()
    cache_payload = {
        "schema_version": FOLD_CACHE_SCHEMA_VERSION,
        "role": role,
        "patch_size": int(patch_size),
        "stride": int(stride),
        "num_inputs": int(num_inputs),
        "samples": _fold_cache_sample_descriptor(samples),
    }
    cache_hash = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(tmpdir, f"tf_cache_{role}_{cache_hash}_{cache_scope}")


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
        train_samples,
        val_samples,
        patch_size,
        stride,
        epochs,
        num_inputs,
        best_batch_size,
        **kwargs,
    ):
        hp_batch_size = best_batch_size
        strategy = tf.distribute.MirroredStrategy()
        global_batch_size = hp_batch_size * strategy.num_replicas_in_sync
        train_cache = build_fold_cache_path(
            train_samples,
            role="train",
            patch_size=patch_size,
            stride=stride,
            num_inputs=num_inputs,
        )
        val_cache = build_fold_cache_path(
            val_samples,
            role="val",
            patch_size=patch_size,
            stride=patch_size,
            num_inputs=num_inputs,
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
        except tf.errors.ResourceExhaustedError:
            print(f"OOM with batch size {hp_batch_size}, marking trial as failure.")
            return {"val_loss": 1e9}

        return {"val_loss": min(history.history["val_loss"])}


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


def create_tuner(
    hypermodel,
    max_trials: int,
    tuning_dir: str,
    run_name: str,
    num_inputs: int,
    validation_fraction: float,
):
    return CVTuner(
        hypermodel=hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        directory=os.path.dirname(
            _tuner_state_dir(
                tuning_dir,
                run_name,
                num_inputs,
                validation_fraction,
            )
        ),
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
    validation_fraction: float,
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
            ("validation_fraction", validation_fraction),
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

    train_samples, val_samples = create_spatial_holdout_split(
        samples,
        tile_size=split_tile_size,
        validation_fraction=validation_fraction,
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

    print_training_image_paths(
        train_samples,
        "Training image order for batch-size probe",
    )

    def dataset_builder_fn(bs):
        global_bs = bs * num_replicas_in_sync
        return build_dataset(
            train_samples,
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
    train_cache = build_fold_cache_path(
        train_samples,
        role="train",
        patch_size=patch_size,
        stride=stride,
        num_inputs=num_inputs,
    )
    val_cache = build_fold_cache_path(
        val_samples,
        role="val",
        patch_size=patch_size,
        stride=patch_size,
        num_inputs=num_inputs,
    )

    if skip_tuning:
        print("Skipping hyperparameter tuning. Using default settings.")
        learning_rate = 1e-3
        selected_hyperparameters = [("learning_rate", learning_rate)]
        print_training_image_paths(
            train_samples,
            "Training image order for final training",
        )
        print_training_image_paths(
            val_samples,
            "Validation image order for final training",
        )

        full_dataset = build_dataset(
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
            validation_fraction=validation_fraction,
        )
        tuning_title = (
            "Resuming tuning"
            if has_saved_tuner_state(
                tuning_dir,
                run_name,
                num_inputs,
                validation_fraction,
            )
            else "Starting tuning"
        )
        print_training_image_paths(
            train_samples,
            "Training image order for tuning",
        )
        print_training_image_paths(
            val_samples,
            "Validation image order for tuning",
        )
        print_section(tuning_title)
        print_key_values(
            [
                ("tuning_dir", tuning_dir),
                ("run_name", run_name),
                ("max_trials", max_trials),
                ("tune_epochs", tune_epochs),
                ("validation_fraction", validation_fraction),
                ("per_replica_batch_size", best_batch_size),
                ("global_batch_size", global_batch_size),
            ]
        )

        tuner.search(
            train_samples=train_samples,
            val_samples=val_samples,
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

        print_training_image_paths(
            train_samples,
            "Training image order for final training",
        )
        print_training_image_paths(
            val_samples,
            "Validation image order for final training",
        )
        full_dataset = build_dataset(
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

        monitor_metric = "val_loss"
        fit_kwargs = {"validation_data": val_dataset}

    log_dir = os.path.join(
        "logs", "fit", f"{run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq="epoch"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=20,
        restore_best_weights=True,
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
                "Resume checkpoint already meets or exceeds the configured final "
                "epoch budget. Skipping additional training."
            )
            should_run_final_training = False

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
                early_stopping,
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
