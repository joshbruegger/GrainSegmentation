import os
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import rotate, shift
from sklearn.model_selection import train_test_split
import tensorflow as tf


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


def _augment_sample(
    images: List[np.ndarray], label: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray]:
    if np.random.rand() < 0.5:
        images = [np.fliplr(img) for img in images]
        label = np.fliplr(label)
    if np.random.rand() < 0.5:
        images = [np.flipud(img) for img in images]
        label = np.flipud(label)

    # Continuous rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(0, 360)
        images = [rotate(img, angle, axes=(0, 1), reshape=False, order=1, mode='reflect') for img in images]
        label = rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode='reflect')

    # Translation
    if np.random.rand() < 0.5:
        shift_y = np.random.uniform(-0.1, 0.1) * label.shape[0]
        shift_x = np.random.uniform(-0.1, 0.1) * label.shape[1]
        images = [shift(img, (shift_y, shift_x, 0), order=1, mode='reflect') for img in images]
        label = shift(label, (shift_y, shift_x), order=0, mode='reflect')

    delta = np.random.uniform(-0.2, 0.2)
    contrast = np.random.uniform(0.8, 1.2)
    augmented = []
    for img in images:
        img = np.clip(img + delta, 0.0, 1.0)
        img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)
        augmented.append(img)
    return augmented, label


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
    img1_suffix: str,
    img_suffix_template: str,
    mask_ext: str | None,
    mask_stem_suffix: str,
    num_inputs: int,
) -> List[Dict[str, Any]]:
    img1_pattern = os.path.join(image_dir, f"*{img1_suffix}.*")
    img1_paths = sorted(glob(img1_pattern))
    if not img1_paths:
        raise ValueError(f"No images found for pattern: {img1_pattern}")

    samples = []
    for img1_path in img1_paths:
        base_name = os.path.basename(img1_path)
        stem, _ = os.path.splitext(base_name)
        if img1_suffix not in stem:
            continue
        base_stem = stem.replace(img1_suffix, "")
        image_paths = []
        for idx in range(1, num_inputs + 1):
            suffix = img_suffix_template.format(index=idx)
            img_path = img1_path.replace(img1_suffix, suffix)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image for input {idx}: {img_path}")
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
    augment: bool,
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

                if augment:
                    patch_images, patch_label = _augment_sample(
                        patch_images, patch_label
                    )

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
        lambda: _sample_patch_generator(
            samples, patch_size, stride, augment, num_inputs
        ),
        output_signature=output_signature,
    )
    if augment:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
