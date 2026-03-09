import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
import warnings
from data import build_dataset, list_samples, create_spatial_cv_folds
from model import build_unet, initialize_from_checkpoint, weighted_crossentropy

warnings.filterwarnings("ignore", message="Your input ran out of data")


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
    print(f"Number of GPUs: {num_gpus}")
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

    best_batch_size = find_optimal_batch_size(
        compiled_model_builder, dataset_builder_fn, start_batch_size=32
    )
    print(f"Optimal per-replica batch size found: {best_batch_size}")
    global_batch_size = best_batch_size * num_replicas_in_sync

    if skip_tuning:
        print("Skipping hyperparameter tuning. Using default settings.")
        train_samples, val_samples = cv_folds[0]
        learning_rate = 1e-3

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

        all_train_samples = []
        for _, val_samples_fold in cv_folds:
            all_train_samples.extend(val_samples_fold)

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

    import datetime

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
    if resume_path:
        initial_epoch = infer_initial_epoch(final_model, full_dataset)

    final_model.fit(
        full_dataset,
        epochs=final_epochs,
        initial_epoch=initial_epoch,
        callbacks=[reduce_lr, checkpoint_best, checkpoint_latest, tensorboard_callback],
        verbose=2,
        **fit_kwargs,
    )

    final_model.save(output_model_path)
    return final_model
