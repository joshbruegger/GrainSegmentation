import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "stack_tiff_channels.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"Missing script under test: {SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("stack_tiff_channels", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StackTiffChannelsTests(unittest.TestCase):
    def test_cli_stacks_rgb_tiffs_into_channel_first_uint8_output(self) -> None:
        module = _load_module()

        first = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint16,
        )
        second = np.array(
            [
                [[21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32]],
            ],
            dtype=np.uint16,
        )

        expected = np.concatenate(
            [
                np.transpose(first.astype(np.uint8), (2, 0, 1)),
                np.transpose(second.astype(np.uint8), (2, 0, 1)),
            ],
            axis=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = tmp_path / "stacked_output.tiff"

            tifffile.imwrite(input_dir / "b_second.tiff", second)
            tifffile.imwrite(input_dir / "a_first.tif", first)

            argv = [
                "stack_tiff_channels.py",
                str(input_dir),
                str(output_path),
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            self.assertTrue(output_path.exists())

            result = tifffile.imread(output_path)
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, (6, 2, 2))
            np.testing.assert_array_equal(result, expected)

    def test_stack_tiff_channels_rejects_non_rgb_stacks(self) -> None:
        module = _load_module()

        non_rgb_stack = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = tmp_path / "stacked_output.tif"

            tifffile.imwrite(
                input_dir / "not_rgb.tif",
                non_rgb_stack,
                photometric="minisblack",
            )

            with self.assertRaisesRegex(ValueError, "RGB"):
                module.stack_tiff_channels(input_dir, output_path)

    def test_stack_tiff_channels_rejects_mismatched_dimensions(self) -> None:
        module = _load_module()

        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.zeros((3, 2, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = tmp_path / "stacked_output.tif"

            tifffile.imwrite(input_dir / "a_first.tif", first)
            tifffile.imwrite(input_dir / "b_second.tif", second)

            with self.assertRaisesRegex(ValueError, "matching height and width"):
                module.stack_tiff_channels(input_dir, output_path)

    def test_stack_tiff_channels_rejects_invalid_output_suffix(self) -> None:
        module = _load_module()

        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = tmp_path / "stacked_output.png"

            tifffile.imwrite(input_dir / "a_first.tif", image)

            with self.assertRaisesRegex(ValueError, r"\.tif or \.tiff"):
                module.stack_tiff_channels(input_dir, output_path)

    def test_stack_tiff_channels_rejects_missing_input_directory(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_input_dir = tmp_path / "missing"
            output_path = tmp_path / "stacked_output.tif"

            with self.assertRaisesRegex(ValueError, "Input directory does not exist"):
                module.stack_tiff_channels(missing_input_dir, output_path)

    def test_stack_tiff_channels_rejects_empty_input_directory(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = tmp_path / "stacked_output.tif"

            with self.assertRaisesRegex(ValueError, "No TIFF files found"):
                module.stack_tiff_channels(input_dir, output_path)

    def test_stack_tiff_channels_ignores_output_file_inside_input_directory(
        self,
    ) -> None:
        module = _load_module()

        first = np.full((2, 2, 3), fill_value=10, dtype=np.uint8)
        second = np.full((2, 2, 3), fill_value=20, dtype=np.uint8)
        ignored_output_seed = np.full((2, 2, 3), fill_value=99, dtype=np.uint8)

        expected = np.concatenate(
            [
                np.transpose(first, (2, 0, 1)),
                np.transpose(second, (2, 0, 1)),
            ],
            axis=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            output_path = input_dir / "z_output.tif"

            tifffile.imwrite(input_dir / "a_first.tif", first)
            tifffile.imwrite(input_dir / "b_second.tif", second)
            tifffile.imwrite(output_path, ignored_output_seed)

            module.stack_tiff_channels(input_dir, output_path)

            result = tifffile.imread(output_path)
            np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
