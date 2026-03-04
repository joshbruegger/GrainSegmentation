import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam

from .data import build_dataset, list_samples, create_spatial_cv_folds
from .model import build_unet, initialize_from_checkpoint, weighted_crossentropy


class CVTuner(kt.BayesianOptimization):
    def run_trial(
        self,
        trial,
        cv_folds,
        patch_size,
        stride,
        batch_size,
        epochs,
        num_inputs,
        **kwargs,
    ):
        val_losses = []
        for i, (train_samples, val_samples) in enumerate(cv_folds):
            train_dataset = build_dataset(
                train_samples,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                augment=True,
                num_inputs=num_inputs,
            )
            val_dataset = build_dataset(
                val_samples,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                augment=False,
                num_inputs=num_inputs,
            )

            model = self.hypermodel.build(trial.hyperparameters)
            learning_rate = trial.hyperparameters.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
            )
            optimizer = Adam(learning_rate=learning_rate)

            if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

            model.compile(
                optimizer=optimizer, loss=weighted_crossentropy, metrics=["accuracy"]
            )

            history = model.fit(
                train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1
            )
            val_losses.append(min(history.history["val_loss"]))

        return {"val_loss": np.mean(val_losses)}


def train_model(
    image_dir: str,
    mask_dir: str,
    checkpoint_path: str | None,
    resume_path: str | None,
    output_model_path: str,
    patch_size: int,
    stride: int,
    batch_size: int,
    epochs: int,
    img1_suffix: str,
    img_suffix_template: str,
    mask_ext: str | None,
    mask_stem_suffix: str,
    split_tile_size: int,
    split_coverage_bins: int,
    num_inputs: int,
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
        img1_suffix=img1_suffix,
        img_suffix_template=img_suffix_template,
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

    tuner = CVTuner(
        hypermodel=lambda hp: build_unet(
            patch_size=patch_size, num_inputs=num_inputs, hp=hp
        ),
        objective="val_loss",
        max_trials=max_trials,
        directory="tuning_dir",
        project_name="unet_tuning",
        overwrite=True,
    )

    tuner.search(
        cv_folds=cv_folds,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        epochs=epochs,
        num_inputs=num_inputs,
    )

    best_hp = tuner.get_best_hyperparameters()[0]
    print(f"Best hyperparameters: {best_hp.values}")

    all_train_samples = []
    for _, val_samples in cv_folds:
        all_train_samples.extend(val_samples)

    full_dataset = build_dataset(
        all_train_samples,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
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
    final_model.fit(full_dataset, epochs=epochs)

    final_model.save(output_model_path)
    return final_model
