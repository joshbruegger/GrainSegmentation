import os
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from sklearn.model_selection import train_test_split
import tensorflow as tf


def build_label_mask(grain_mask: np.ndarray) -> np.ndarray:
    if grain_mask.ndim != 2:
        raise ValueError("grain_mask must be 2D")
    grain = grain_mask.astype(bool)
    eroded = binary_erosion(grain, iterations=1, border_value=0)
    boundary = grain & ~eroded

    label = np.zeros_like(grain_mask, dtype=np.uint8)
    label[grain] = 1
    label[boundary] = 2
    return label


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


def split_samples(
    samples: List[Dict[str, Any]],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(samples) < 3:
        return samples, [], []

    train_val, test = train_test_split(
        samples, test_size=test_size, random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state
    )
    return train, val, test


def _grid_regions(height: int, width: int, tile_size: int) -> List[Tuple[int, int, int, int]]:
    regions = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            regions.append((y0, y1, x0, x1))
    return regions


def _split_indices_by_coverage(
    coverages: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
    coverage_bins: int,
) -> Tuple[List[int], List[int], List[int]]:
    if coverage_bins < 1:
        raise ValueError("coverage_bins must be >= 1")
    rng = np.random.default_rng(random_state)
    total = coverages.shape[0]
    if total == 0:
        return [], [], []
    if total == 1:
        return [0], [], []

    bins = min(coverage_bins, total)
    bin_edges = np.quantile(coverages, np.linspace(0.0, 1.0, bins + 1))
    if np.allclose(bin_edges, bin_edges[0]):
        bin_ids = np.zeros(total, dtype=int)
    else:
        bin_ids = np.digitize(coverages, bin_edges[1:-1], right=True)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for bin_id in range(bin_ids.max() + 1):
        indices = np.where(bin_ids == bin_id)[0]
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        n = indices.size
        n_val = int(round(val_size * n))
        n_test = int(round(test_size * n))
        if n_val + n_test >= n:
            overflow = n_val + n_test - n
            if n_test >= overflow:
                n_test -= overflow
            else:
                overflow -= n_test
                n_test = 0
                n_val = max(0, n_val - overflow)

        val_idx.extend(indices[:n_val].tolist())
        test_idx.extend(indices[n_val : n_val + n_test].tolist())
        train_idx.extend(indices[n_val + n_test :].tolist())

    return train_idx, val_idx, test_idx


def split_samples_spatial(
    samples: List[Dict[str, Any]],
    tile_size: int,
    test_size: float,
    val_size: float,
    random_state: int,
    coverage_bins: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for sample in samples:
        mask = _load_raster_mask(sample["mask"])
        height, width = mask.shape
        regions = _grid_regions(height, width, tile_size)
        if not regions:
            continue
        coverages = np.array(
            [
                float(np.mean(mask[y0:y1, x0:x1] > 0))
                for (y0, y1, x0, x1) in regions
            ],
            dtype=np.float32,
        )
        train_idx, val_idx, test_idx = _split_indices_by_coverage(
            coverages,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            coverage_bins=coverage_bins,
        )

        for idx in train_idx:
            region_sample = dict(sample)
            region_sample["region"] = regions[idx]
            train_samples.append(region_sample)
        for idx in val_idx:
            region_sample = dict(sample)
            region_sample["region"] = regions[idx]
            val_samples.append(region_sample)
        for idx in test_idx:
            region_sample = dict(sample)
            region_sample["region"] = regions[idx]
            test_samples.append(region_sample)

    return train_samples, val_samples, test_samples


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
        grain_mask = (raw_mask > 0).astype(np.uint8)
        region = sample.get("region")
        if region is not None:
            y0, y1, x0, x1 = region
            images = [img[y0:y1, x0:x1, :] for img in images]
            grain_mask = grain_mask[y0:y1, x0:x1]
        label = build_label_mask(grain_mask)

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
