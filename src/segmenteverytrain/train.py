import tensorflow as tf
from keras.optimizers import Adam

from .data import build_dataset, list_samples, split_samples, split_samples_spatial
from .model import build_unet, initialize_from_checkpoint, weighted_crossentropy


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
    spatial_split: bool,
    split_tile_size: int,
    split_coverage_bins: int,
    num_inputs: int,
    test_size: float,
    val_size: float,
    random_state: int,
    use_mixed_precision: bool,
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
    if spatial_split:
        if split_tile_size < patch_size:
            raise ValueError("split_tile_size must be >= patch_size")
        train_samples, val_samples, test_samples = split_samples_spatial(
            samples,
            tile_size=split_tile_size,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            coverage_bins=split_coverage_bins,
        )
    else:
        train_samples, val_samples, test_samples = split_samples(
            samples, test_size=test_size, val_size=val_size, random_state=random_state
        )
    if not train_samples:
        raise ValueError("No training samples found.")

    train_dataset = build_dataset(
        train_samples,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        augment=True,
        num_inputs=num_inputs,
    )
    val_dataset = (
        build_dataset(
            val_samples,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            augment=False,
            num_inputs=num_inputs,
        )
        if val_samples
        else None
    )
    test_dataset = (
        build_dataset(
            test_samples,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            augment=False,
            num_inputs=num_inputs,
        )
        if test_samples
        else None
    )

    if checkpoint_path:
        model = initialize_from_checkpoint(
            checkpoint_path, patch_size=patch_size, num_inputs=num_inputs
        )
    elif resume_path:
        model = tf.keras.models.load_model(
            resume_path, custom_objects={"weighted_crossentropy": weighted_crossentropy}
        )
    else:
        model = build_unet(patch_size=patch_size, num_inputs=num_inputs)

    optimizer = Adam()
    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(optimizer=optimizer, loss=weighted_crossentropy, metrics=["accuracy"])

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset if val_dataset is not None else None,
    )

    if test_dataset is not None:
        model.evaluate(test_dataset)

    model.save(output_model_path)
    return model
