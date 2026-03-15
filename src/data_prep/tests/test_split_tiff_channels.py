import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "split_tiff_channels.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"Missing script under test: {SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("split_tiff_channels", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitTiffChannelsTests(unittest.TestCase):
    def test_cli_splits_channel_first_stack_into_rgb_tiffs(self) -> None:
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
        stacked = np.concatenate(
            [
                np.transpose(first.astype(np.uint8), (2, 0, 1)),
                np.transpose(second.astype(np.uint8), (2, 0, 1)),
            ],
            axis=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "stacked_output.tif"
            output_dir = tmp_path / "split_outputs"
            tifffile.imwrite(
                input_path,
                stacked,
                photometric="rgb",
                planarconfig="separate",
            )

            argv = [
                "split_tiff_channels.py",
                str(input_path),
                str(output_dir),
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            first_result = tifffile.imread(output_dir / "stacked_output_000.tif")
            second_result = tifffile.imread(output_dir / "stacked_output_001.tif")

            self.assertEqual(first_result.dtype, np.uint8)
            self.assertEqual(second_result.dtype, np.uint8)
            np.testing.assert_array_equal(first_result, first.astype(np.uint8))
            np.testing.assert_array_equal(second_result, second.astype(np.uint8))

    def test_split_tiff_channels_accepts_channel_last_input(self) -> None:
        module = _load_module()

        first = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        second = np.array(
            [
                [[13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24]],
            ],
            dtype=np.uint8,
        )
        stacked = np.concatenate([first, second], axis=-1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "stacked_output.tiff"
            output_dir = tmp_path / "split_outputs"
            tifffile.imwrite(input_path, stacked)

            output_files = module.split_tiff_channels(
                input_path, output_dir, prefix="rgb"
            )

            self.assertEqual(
                output_files,
                [
                    output_dir / "rgb_000.tif",
                    output_dir / "rgb_001.tif",
                ],
            )
            np.testing.assert_array_equal(tifffile.imread(output_files[0]), first)
            np.testing.assert_array_equal(tifffile.imread(output_files[1]), second)

    def test_split_tiff_channels_rejects_non_rgb_channel_count(self) -> None:
        module = _load_module()

        invalid = np.zeros((4, 2, 2), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "invalid_stack.tif"
            output_dir = tmp_path / "split_outputs"
            tifffile.imwrite(input_path, invalid, photometric="minisblack")

            with self.assertRaisesRegex(ValueError, "divisible by 3"):
                module.split_tiff_channels(input_path, output_dir)

    def test_split_tiff_channels_rejects_invalid_input_suffix(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "invalid_stack.png"
            input_path.write_bytes(b"not a tiff")
            output_dir = tmp_path / "split_outputs"

            with self.assertRaisesRegex(ValueError, r"\.tif or \.tiff"):
                module.split_tiff_channels(input_path, output_dir)

    def test_split_tiff_channels_rejects_missing_input_file(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "missing_stack.tif"
            output_dir = tmp_path / "split_outputs"

            with self.assertRaisesRegex(ValueError, "does not exist"):
                module.split_tiff_channels(input_path, output_dir)


if __name__ == "__main__":
    unittest.main()
