import tempfile
import unittest
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

try:
    import tensorflow  # noqa: F401
except ModuleNotFoundError:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.Tensor = object
    tf_stub.int32 = "int32"
    tf_stub.float32 = "float32"
    tf_stub.data = types.SimpleNamespace(Dataset=object, AUTOTUNE=object())
    sys.modules["tensorflow"] = tf_stub

import data


def _write_rgb(path: Path, shape: tuple[int, int]) -> None:
    array = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def _write_float_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.float32), mode="F").save(path)


def _consume_first_batch(sample: dict, patch_size: int, stride: int, num_inputs: int):
    dataset = data.build_dataset(
        samples=[sample],
        patch_size=patch_size,
        stride=stride,
        batch_size=1,
        augment=False,
        num_inputs=num_inputs,
    )
    return next(iter(dataset.take(1)))


class BuildDatasetValidationTests(unittest.TestCase):
    def test_build_dataset_rejects_non_positive_patch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image_path, (2, 2))
            _write_mask(mask_path, np.zeros((2, 2), dtype=np.uint8))

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(
                ValueError, "patch_size and stride must be > 0"
            ):
                data.build_dataset(
                    samples=[sample],
                    patch_size=0,
                    stride=1,
                    batch_size=1,
                    augment=False,
                    num_inputs=1,
                )

    def test_build_dataset_rejects_non_positive_stride(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image_path, (2, 2))
            _write_mask(mask_path, np.zeros((2, 2), dtype=np.uint8))

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(
                ValueError, "patch_size and stride must be > 0"
            ):
                data.build_dataset(
                    samples=[sample],
                    patch_size=2,
                    stride=0,
                    batch_size=1,
                    augment=False,
                    num_inputs=1,
                )

    def test_build_dataset_rejects_mismatched_input_image_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image1_path = tmp_path / "image1.png"
            image2_path = tmp_path / "image2.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image1_path, (2, 2))
            _write_rgb(image2_path, (3, 3))
            _write_mask(mask_path, np.zeros((2, 2), dtype=np.uint8))

            sample = {
                "images": [str(image1_path), str(image2_path)],
                "mask": str(mask_path),
            }

            with self.assertRaisesRegex(
                Exception, "All input images must share the same shape"
            ):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=2)

    def test_build_dataset_rejects_mask_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image_path, (2, 2))
            _write_mask(mask_path, np.zeros((3, 3), dtype=np.uint8))

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(
                Exception, "Mask shape .* does not match image shape"
            ):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=1)

    def test_build_dataset_rejects_invalid_mask_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image_path, (2, 2))
            _write_mask(mask_path, np.array([[0, 1], [2, 3]], dtype=np.uint8))

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(Exception, "Mask values must be in \\[0, 2\\]"):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=1)

    def test_build_dataset_rejects_float_mask_values_outside_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.tiff"
            _write_rgb(image_path, (2, 2))
            _write_float_mask(
                mask_path, np.array([[0.0, 1.2], [2.0, 2.9]], dtype=np.float32)
            )

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(Exception, "Mask values must be in \\[0, 2\\]"):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=1)

    def test_build_dataset_rejects_near_integer_float_mask_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.tiff"
            _write_rgb(image_path, (2, 2))
            _write_float_mask(
                mask_path,
                np.array([[0.0, 1.0], [2.0, 1.000001]], dtype=np.float32),
            )

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(Exception, "Mask values must be in \\[0, 2\\]"):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=1)

    def test_build_dataset_rejects_num_inputs_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            _write_rgb(image_path, (2, 2))
            _write_mask(mask_path, np.zeros((2, 2), dtype=np.uint8))

            sample = {"images": [str(image_path)], "mask": str(mask_path)}

            with self.assertRaisesRegex(
                Exception, "Mismatch between num_inputs and loaded images"
            ):
                _consume_first_batch(sample, patch_size=2, stride=2, num_inputs=2)

    def test_build_dataset_rejects_mixed_sample_image_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image1_path = tmp_path / "image1.png"
            image2_path = tmp_path / "image2.png"
            image3_path = tmp_path / "image3.png"
            mask1_path = tmp_path / "mask1.png"
            mask2_path = tmp_path / "mask2.png"
            _write_rgb(image1_path, (2, 2))
            _write_rgb(image2_path, (2, 2))
            _write_rgb(image3_path, (2, 2))
            _write_mask(mask1_path, np.zeros((2, 2), dtype=np.uint8))
            _write_mask(mask2_path, np.zeros((2, 2), dtype=np.uint8))

            samples = [
                {
                    "images": [str(image1_path), str(image2_path)],
                    "mask": str(mask1_path),
                },
                {"images": [str(image3_path)], "mask": str(mask2_path)},
            ]

            with self.assertRaisesRegex(
                ValueError, "Mismatch between num_inputs and loaded images"
            ):
                data.build_dataset(
                    samples=samples,
                    patch_size=2,
                    stride=2,
                    batch_size=1,
                    augment=False,
                    num_inputs=2,
                )


class SpatialHoldoutSplitTests(unittest.TestCase):
    def _create_region_samples(self, tmp_path: Path) -> list[dict]:
        samples = []
        for idx, values in enumerate(
            (
                np.array(
                    [
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [2, 2, 0, 0],
                        [2, 2, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
                np.array(
                    [
                        [0, 1, 1, 2],
                        [0, 1, 1, 2],
                        [0, 0, 2, 2],
                        [0, 0, 2, 2],
                    ],
                    dtype=np.uint8,
                ),
            ),
            start=1,
        ):
            image_path = tmp_path / f"sample{idx}_PPL.png"
            mask_path = tmp_path / f"sample{idx}_labels.png"
            _write_rgb(image_path, (4, 4))
            _write_mask(mask_path, values)
            samples.append(
                {
                    "id": f"sample-{idx}",
                    "images": [str(image_path)],
                    "mask": str(mask_path),
                }
            )
        return samples

    def test_create_spatial_holdout_split_is_deterministic_for_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = self._create_region_samples(Path(tmpdir))

            first = data.create_spatial_holdout_split(
                samples=samples,
                tile_size=2,
                validation_fraction=0.25,
                random_state=42,
                coverage_bins=4,
            )
            second = data.create_spatial_holdout_split(
                samples=samples,
                tile_size=2,
                validation_fraction=0.25,
                random_state=42,
                coverage_bins=4,
            )

        self.assertEqual(first, second)

    def test_create_spatial_holdout_split_returns_disjoint_complete_partition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = self._create_region_samples(Path(tmpdir))
            train_samples, val_samples = data.create_spatial_holdout_split(
                samples=samples,
                tile_size=2,
                validation_fraction=0.25,
                random_state=7,
                coverage_bins=4,
            )

        train_regions = {
            (sample["id"], tuple(sample["region"])) for sample in train_samples
        }
        val_regions = {
            (sample["id"], tuple(sample["region"])) for sample in val_samples
        }

        self.assertFalse(train_regions & val_regions)
        self.assertEqual(len(train_regions) + len(val_regions), 8)

    def test_create_spatial_holdout_split_reserves_requested_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = self._create_region_samples(Path(tmpdir))
            train_samples, val_samples = data.create_spatial_holdout_split(
                samples=samples,
                tile_size=2,
                validation_fraction=0.25,
                random_state=1,
                coverage_bins=4,
            )

        self.assertEqual(len(val_samples), 2)
        self.assertEqual(len(train_samples), 6)

    def test_create_spatial_holdout_split_keeps_low_coverage_regions_in_train(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            samples = []
            masks = [
                np.full((10, 10), 1, dtype=np.uint8),
                np.full((10, 10), 2, dtype=np.uint8),
                np.pad(
                    np.array([[1]], dtype=np.uint8),
                    pad_width=((0, 9), (0, 9)),
                    mode="constant",
                    constant_values=0,
                ),
            ]

            for idx, mask_values in enumerate(masks, start=1):
                image_path = tmp_path / f"holdout{idx}_PPL.png"
                mask_path = tmp_path / f"holdout{idx}_labels.png"
                _write_rgb(image_path, (10, 10))
                _write_mask(mask_path, mask_values)
                samples.append(
                    {
                        "id": f"holdout-{idx}",
                        "images": [str(image_path)],
                        "mask": str(mask_path),
                    }
                )

            train_samples, val_samples = data.create_spatial_holdout_split(
                samples=samples,
                tile_size=10,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
            )

        train_ids = {sample["id"] for sample in train_samples}
        val_ids = {sample["id"] for sample in val_samples}

        self.assertIn("holdout-3", train_ids)
        self.assertNotIn("holdout-3", val_ids)
        self.assertEqual(len(val_samples), 1)


if __name__ == "__main__":
    unittest.main()
