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
        tf.keras.backend.clear_session()
        try:
            model = model_builder()
            dataset = train_dataset_fn(batch_size)
            model.fit(dataset, steps_per_epoch=1, epochs=1, verbose=0)
            print(f"Successfully ran with batch size: {batch_size}")
            tf.keras.backend.clear_session()
            return batch_size
        except tf.errors.ResourceExhaustedError:
            print(f"OOM with batch size {batch_size}. Halving batch size.")
            batch_size //= 2

    raise RuntimeError(
        "Could not find a batch size that fits in memory, even batch_size=1 failed."
    )


class CVTuner(kt.BayesianOptimization):
    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def run_trial(
        self,
        trial,
        cv_folds,
        patch_size,
        stride,
        epochs,
        num_inputs,
        **kwargs,
    ):
        hp_batch_size = trial.hyperparameters.Choice("batch_size", [1, 2, 4, 8, 16, 32])
        global_batch_size = hp_batch_size * self.strategy.num_replicas_in_sync

        val_losses = []
        for i, (train_samples, val_samples) in enumerate(cv_folds):
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

            with self.strategy.scope():
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
                tf.keras.backend.clear_session()
                return {"val_loss": 1e9}

        return {"val_loss": np.mean(val_losses)}


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

    strategy = tf.distribute.MirroredStrategy()

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
        if checkpoint_path:
            return initialize_from_checkpoint(
                checkpoint_path, patch_size, num_inputs=num_inputs, hp=hp
            )
        elif resume_path:
            model = tf.keras.models.load_model(
                resume_path,
                custom_objects={"weighted_crossentropy": weighted_crossentropy},
            )
            # We can still tune learning rate if we want
            if hp:
                learning_rate = hp.Float(
                    "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
                )
                from keras.optimizers import Adam

                optimizer = Adam(learning_rate=learning_rate)
                # the tuner compiling happens in run_trial, so we just return the model
            return model
        else:
            if hp is None:
                return build_unet(
                    patch_size, num_inputs=num_inputs, hp=hp, base_filters=16
                )
            return build_unet(patch_size, num_inputs=num_inputs, hp=hp)

    if skip_tuning:
        print("Skipping hyperparameter tuning. Using default settings.")
        train_samples, val_samples = cv_folds[0]
        learning_rate = 1e-3

        def compiled_model_builder():
            with strategy.scope():
                model = build_hypermodel(hp=None)
                if resume_path and getattr(model, "optimizer", None) is not None:
                    print("Resuming from checkpoint, retaining optimizer state.")
                    return model

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss=weighted_crossentropy,
                    metrics=["accuracy"],
                )
                return model

        def dataset_builder_fn(bs):
            global_bs = bs * strategy.num_replicas_in_sync
            return build_dataset(
                train_samples,
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

        global_batch_size = best_batch_size * strategy.num_replicas_in_sync

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
        tuner = CVTuner(
            strategy=strategy,
            hypermodel=build_hypermodel,
            objective="val_loss",
            max_trials=max_trials,
            directory=os.path.join(tuning_dir, f"tuning_{run_name}_{num_inputs}in"),
            project_name="unet_tuning",
            overwrite=True,
        )

        tuner.search(
            cv_folds=cv_folds,
            patch_size=patch_size,
            stride=stride,
            epochs=tune_epochs,
            num_inputs=num_inputs,
        )

        best_hp = tuner.get_best_hyperparameters()[0]
        print(f"Best hyperparameters: {best_hp.values}")

        best_batch_size = best_hp.get("batch_size")
        global_batch_size = best_batch_size * strategy.num_replicas_in_sync
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

        with strategy.scope():
            final_model = tuner.hypermodel.build(best_hp)
            optimizer = Adam(learning_rate=learning_rate)
            final_model.compile(
                optimizer=optimizer, loss=weighted_crossentropy, metrics=["accuracy"]
            )

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
    if (
        skip_tuning
        and resume_path
        and getattr(final_model, "optimizer", None) is not None
    ):
        if final_model.optimizer.iterations.numpy() > 0:
            print("Calculating dataset length for resuming...")
            steps_per_epoch = 0
            for _ in full_dataset:
                steps_per_epoch += 1
            if steps_per_epoch > 0:
                initial_epoch = int(
                    final_model.optimizer.iterations.numpy() // steps_per_epoch
                )
            print(f"Resuming training from epoch {initial_epoch}")

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
