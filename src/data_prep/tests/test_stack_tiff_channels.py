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


if __name__ == "__main__":
    unittest.main()
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


if __name__ == "__main__":
    unittest.main()
