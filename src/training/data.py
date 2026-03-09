import os
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import rotate, shift
import tensorflow as tf

Image.MAX_IMAGE_PIXELS = None


def _compute_starts_tf(size: tf.Tensor, patch_size: tf.Tensor, stride: tf.Tensor) -> tf.Tensor:
    size = tf.cast(size, tf.int32)
    patch_size = tf.cast(patch_size, tf.int32)
    stride = tf.cast(stride, tf.int32)
    
    def get_starts():
        limit = size - patch_size
        starts = tf.range(0, limit + 1, stride)
        last_start = starts[-1]
        
        return tf.cond(
            tf.equal(last_start, limit),
            lambda: starts,
            lambda: tf.concat([starts, [limit]], axis=0)
        )
        
    return tf.cond(
        size <= patch_size,
        lambda: tf.constant([0], dtype=tf.int32),
        get_starts
    )


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


def build_dataset(
    samples: List[Dict[str, Any]],
    patch_size: int,
    stride: int,
    batch_size: int,
    augment: bool,
    num_inputs: int,
    cache_file: str | None = None,
) -> tf.data.Dataset:
    if not samples:
        raise ValueError("samples list must not be empty")
        
    images_list = [s["images"] for s in samples]
    mask_list = [s["mask"] for s in samples]
    region_list = [s.get("region", (0, 0, 0, 0)) for s in samples]
    has_region_list = [("region" in s) for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((
        images_list, mask_list, region_list, has_region_list
    ))
    
    def _load_sample_py(image_paths_tensor, mask_path_tensor, region_tensor, has_region_tensor):
        image_paths = [p.decode("utf-8") for p in image_paths_tensor.numpy()]
        mask_path = mask_path_tensor.numpy().decode("utf-8")
        region = region_tensor.numpy()
        has_region = has_region_tensor.numpy()

        images = [_load_rgb_image(p) for p in image_paths]
        mask = _load_raster_mask(mask_path)

        if has_region:
            y0, y1, x0, x1 = region
            images = [img[y0:y1, x0:x1, :] for img in images]
            mask = mask[y0:y1, x0:x1]
            
        return tuple(images) + (mask.astype(np.int32),)

    def _py_wrapper(image_paths, mask_path, region, has_region):
        res = tf.py_function(
            func=_load_sample_py,
            inp=[image_paths, mask_path, region, has_region],
            Tout=[tf.float32] * num_inputs + [tf.int32]
        )
        for i in range(num_inputs):
            res[i].set_shape([None, None, 3])
        res[-1].set_shape([None, None])
        return tuple(res[:num_inputs]), res[-1]

    ds = ds.map(_py_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    def pad_fn(images_tuple, mask):
        shape = tf.shape(mask)
        height, width = shape[0], shape[1]
        
        pad_h = tf.maximum(0, patch_size - height)
        pad_w = tf.maximum(0, patch_size - width)
        
        paddings_img = [[0, pad_h], [0, pad_w], [0, 0]]
        paddings_mask = [[0, pad_h], [0, pad_w]]
        
        padded_images = tuple([tf.pad(img, paddings_img, mode='CONSTANT', constant_values=0.0) for img in images_tuple])
        padded_mask = tf.pad(mask, paddings_mask, mode='CONSTANT', constant_values=0)
        
        return padded_images, padded_mask

    ds = ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def extract_patches_fn(images_tuple, mask):
        shape = tf.shape(mask)
        height = shape[0]
        width = shape[1]
        
        y_starts = _compute_starts_tf(height, tf.constant(patch_size, dtype=tf.int32), tf.constant(stride, dtype=tf.int32))
        x_starts = _compute_starts_tf(width, tf.constant(patch_size, dtype=tf.int32), tf.constant(stride, dtype=tf.int32))
        
        Y, X = tf.meshgrid(y_starts, x_starts, indexing='ij')
        coords = tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])], axis=1)
        
        coord_ds = tf.data.Dataset.from_tensor_slices(coords)
        
        def crop_patch(coord):
            y = coord[0]
            x = coord[1]
            
            patch_images = tuple([
                tf.image.crop_to_bounding_box(img, y, x, patch_size, patch_size)
                for img in images_tuple
            ])
            patch_mask = tf.image.crop_to_bounding_box(tf.expand_dims(mask, -1), y, x, patch_size, patch_size)
            patch_mask = tf.squeeze(patch_mask, axis=-1)
            
            patch_label = tf.one_hot(patch_mask, depth=3, dtype=tf.float32)
            
            return patch_images, patch_label
            
        return coord_ds.map(crop_patch)

    ds = ds.flat_map(extract_patches_fn)

    if cache_file:
        ds = ds.cache(cache_file)
    else:
        ds = ds.cache()

    if augment:
        ds = ds.map(
            _tf_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        ds = ds.shuffle(buffer_size=100)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
