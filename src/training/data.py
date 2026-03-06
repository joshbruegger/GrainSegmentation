import os
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import rotate, shift
import tensorflow as tf

Image.MAX_IMAGE_PIXELS = None


def _pad_to_patch(
    images: List[np.ndarray], label: np.ndarray, patch_size: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    height, width = label.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h == 0 and pad_w == 0:
        return images, label

    pad_y = (0, pad_h)
    pad_x = (0, pad_w)
    label = np.pad(label, (pad_y, pad_x), mode="constant", constant_values=0)

    padded_images = []
    for img in images:
        padded = np.pad(img, (pad_y, pad_x, (0, 0)), mode="constant", constant_values=0)
        padded_images.append(padded)
    return padded_images, label


def _compute_starts(size: int, patch_size: int, stride: int) -> List[int]:
    if size <= patch_size:
        return [0]
    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] != size - patch_size:
        starts.append(size - patch_size)
    return starts


def _tf_augment(
    inputs: Tuple[tf.Tensor, ...], label: tf.Tensor
) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
    num_inputs = len(inputs)
    # Concatenate all inputs and label along channel axis to apply same spatial transforms
    concat = tf.concat(list(inputs) + [label], axis=-1)

    seed_lr = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    concat = tf.cond(
        seed_lr < 0.5, lambda: tf.image.flip_left_right(concat), lambda: concat
    )

    seed_ud = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    concat = tf.cond(
        seed_ud < 0.5, lambda: tf.image.flip_up_down(concat), lambda: concat
    )

    # Random 90-degree rotations
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    concat = tf.image.rot90(concat, k=k)

    # Split back
    split_sizes = [3] * num_inputs + [3]
    splits = tf.split(concat, split_sizes, axis=-1)

    aug_inputs = list(splits[:-1])
    aug_label = splits[-1]

    # Color augmentations on images only
    final_inputs = []
    for img in aug_inputs:
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

        # Explicit clipping matching the original checkpoint code
        img = tf.where(img < 0.0, tf.zeros_like(img), img)
        img = tf.where(img > 1.0, tf.ones_like(img), img)
        final_inputs.append(img)

    return tuple(final_inputs), aug_label


def _load_rgb_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0


def _load_raster_mask(path: str) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode not in ("L", "I", "I;16", "F"):
            img = img.convert("L")
        mask = np.asarray(img)
    if mask.ndim != 2:
        raise ValueError(f"Raster mask must be 2D: {path}")
    return mask


def list_samples(
    image_dir: str,
    mask_dir: str,
    image_suffixes: List[str],
    mask_ext: str | None,
    mask_stem_suffix: str,
    num_inputs: int,
) -> List[Dict[str, Any]]:
    if not image_suffixes:
        raise ValueError("image_suffixes must not be empty")

    img1_suffix = image_suffixes[0]
    img1_pattern = os.path.join(image_dir, f"*{img1_suffix}.*")
    img1_paths = sorted(glob(img1_pattern))
    if not img1_paths:
        raise ValueError(f"No images found for pattern: {img1_pattern}")

    samples = []
    for img1_path in img1_paths:
        base_name = os.path.basename(img1_path)
        stem, _ = os.path.splitext(base_name)
        if not stem.endswith(img1_suffix):
            continue
        base_stem = stem[: -len(img1_suffix)]
        image_paths = []
        for idx, suffix in enumerate(image_suffixes[:num_inputs]):
            img_ext = os.path.splitext(img1_path)[1]
            img_path = os.path.join(
                os.path.dirname(img1_path), f"{base_stem}{suffix}{img_ext}"
            )
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Missing image for input {idx + 1} ({suffix}): {img_path}"
                )
            image_paths.append(img_path)

        sample = {"images": image_paths, "id": base_stem}
        if mask_ext is None:
            mask_exts = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
        else:
            mask_exts = [mask_ext]
        mask_path = None
        for ext in mask_exts:
            candidate = os.path.join(mask_dir, f"{base_stem}{mask_stem_suffix}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(
                f"Missing raster mask for {base_stem} in {mask_dir}"
            )
        sample["mask"] = mask_path

        samples.append(sample)
    return samples


def _grid_regions(
    height: int, width: int, tile_size: int
) -> List[Tuple[int, int, int, int]]:
    regions = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            regions.append((y0, y1, x0, x1))
    return regions


def create_spatial_cv_folds(
    samples: List[Dict[str, Any]],
    tile_size: int,
    n_splits: int,
    random_state: int,
    coverage_bins: int,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    from sklearn.model_selection import StratifiedKFold, KFold

    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    region_samples: List[Dict[str, Any]] = []
    coverages: List[float] = []

    for sample in samples:
        mask = _load_raster_mask(sample["mask"])
        height, width = mask.shape
        regions = _grid_regions(height, width, tile_size)
        if not regions:
            continue
        for y0, y1, x0, x1 in regions:
            cov = float(np.mean(mask[y0:y1, x0:x1] > 0))
            coverages.append(cov)

            region_sample = dict(sample)
            region_sample["region"] = (y0, y1, x0, x1)
            region_samples.append(region_sample)

    coverages_arr = np.array(coverages, dtype=np.float32)
    total = coverages_arr.shape[0]

    if total < n_splits:
        raise ValueError(f"Not enough regions ({total}) for {n_splits} splits.")

    bins = min(coverage_bins, max(1, total // n_splits))
    bin_edges = np.quantile(coverages_arr, np.linspace(0.0, 1.0, bins + 1))
    if np.allclose(bin_edges, bin_edges[0]):
        bin_ids = np.zeros(total, dtype=int)
    else:
        bin_ids = np.digitize(coverages_arr, bin_edges[1:-1], right=True)

    counts = np.bincount(bin_ids)
    if counts.min() < n_splits:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(np.zeros(total))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(np.zeros(total), bin_ids)

    folds = []
    for train_idx, val_idx in splits:
        train_samples_fold = [region_samples[i] for i in train_idx]
        val_samples_fold = [region_samples[i] for i in val_idx]
        folds.append((train_samples_fold, val_samples_fold))

    return folds


def _sample_patch_generator(
    samples: Iterable[Dict[str, Any]],
    patch_size: int,
    stride: int,
    num_inputs: int,
) -> Iterable[Tuple[Tuple[np.ndarray, ...], np.ndarray]]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be > 0")
    for sample in samples:
        images = [_load_rgb_image(p) for p in sample["images"]]
        if len(images) != num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")
        height, width, _ = images[0].shape
        for img in images[1:]:
            if img.shape != images[0].shape:
                raise ValueError("All input images must share the same shape.")

        raw_mask = _load_raster_mask(sample["mask"])
        if raw_mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {raw_mask.shape} does not match image "
                f"shape {(height, width)} for {sample['mask']}"
            )
        label = raw_mask.astype(np.uint8)
        region = sample.get("region")
        if region is not None:
            y0, y1, x0, x1 = region
            images = [img[y0:y1, x0:x1, :] for img in images]
            label = label[y0:y1, x0:x1]

        images, label = _pad_to_patch(images, label, patch_size)
        height, width = label.shape

        y_starts = _compute_starts(height, patch_size, stride)
        x_starts = _compute_starts(width, patch_size, stride)

        for y in y_starts:
            for x in x_starts:
                patch_images = [
                    img[y : y + patch_size, x : x + patch_size, :] for img in images
                ]
                patch_label = label[y : y + patch_size, x : x + patch_size]

                one_hot = np.eye(3, dtype=np.float32)[patch_label]
                yield tuple(patch_images[:num_inputs]), one_hot


def build_dataset(
    samples: List[Dict[str, Any]],
    patch_size: int,
    stride: int,
    batch_size: int,
    augment: bool,
    num_inputs: int,
) -> tf.data.Dataset:
    image_spec = tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32)
    output_signature = (
        tuple([image_spec] * num_inputs),
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _sample_patch_generator(samples, patch_size, stride, num_inputs),
        output_signature=output_signature,
    )

    dataset = dataset.cache()

    if augment:
        dataset = dataset.map(
            _tf_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
