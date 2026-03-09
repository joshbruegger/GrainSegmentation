import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

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


if __name__ == "__main__":
    unittest.main()
