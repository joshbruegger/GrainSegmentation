import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore", message="Your input ran out of data")

from .data import build_dataset, list_samples, create_spatial_cv_folds
from .model import build_unet, initialize_from_checkpoint, weighted_crossentropy


class CVTuner(kt.BayesianOptimization):
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

        val_losses = []
        for i, (train_samples, val_samples) in enumerate(cv_folds):
            train_dataset = build_dataset(
                train_samples,
                patch_size=patch_size,
                stride=stride,
                batch_size=hp_batch_size,
                augment=True,
                num_inputs=num_inputs,
            )
            val_dataset = build_dataset(
                val_samples,
                patch_size=patch_size,
                stride=patch_size,  # Validation stride = patch_size for 0% overlap
                batch_size=hp_batch_size,
                augment=False,
                num_inputs=num_inputs,
            )

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

            if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

            model.compile(
                optimizer=optimizer, loss=weighted_crossentropy, metrics=["accuracy"]
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
) -> tf.keras.Model:
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

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

    def build_hypermodel(hp):
        if checkpoint_path:
            return initialize_from_checkpoint(
                checkpoint_path, patch_size, num_inputs=num_inputs, hp=hp
            )
        elif resume_path:
            tf.keras.backend.clear_session()
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
            return build_unet(patch_size, num_inputs=num_inputs, hp=hp)

    tuner = CVTuner(
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

    all_train_samples = []
    for _, val_samples in cv_folds:
        all_train_samples.extend(val_samples)

    full_dataset = build_dataset(
        all_train_samples,
        patch_size=patch_size,
        stride=stride,
        batch_size=best_batch_size,
        augment=True,
        num_inputs=num_inputs,
    )

    final_model = tuner.hypermodel.build(best_hp)
    optimizer = Adam(learning_rate=best_hp.get("learning_rate"))
    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    final_model.compile(
        optimizer=optimizer, loss=weighted_crossentropy, metrics=["accuracy"]
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_model_path.replace(".keras", "_best.keras"),
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )

    final_model.fit(
        full_dataset, epochs=final_epochs, callbacks=[reduce_lr, checkpoint], verbose=2
    )

    final_model.save(output_model_path)
    return final_model
