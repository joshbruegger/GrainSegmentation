import importlib.util
import io
import unittest
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FILTER_PATH = REPO_ROOT / "SLURM" / "filter_tensorflow_stderr.py"


def _load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "filter_tensorflow_stderr", FILTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {FILTER_PATH}")

    module = importlib.util.module_from_spec(spec)
    with patch("sys.stdin", io.StringIO("")):
        spec.loader.exec_module(module)
    return module


class FilterTensorflowStderrTests(unittest.TestCase):
    def test_should_suppress_known_tensorflow_noise(self) -> None:
        module = _load_filter_module()

        self.assertTrue(
            module.should_suppress(
                "Unable to register cuFFT factory: Attempting to register factory"
            )
        )
        self.assertTrue(
            module.should_suppress(
                "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR"
            )
        )
        self.assertTrue(
            module.should_suppress(
                "W0000 00:00:1772929499.424729 2575437 gpu_timer.cc:114] Skipping the delay kernel, measurement accuracy will be reduced"
            )
        )

    def test_should_suppress_known_ptx85_feature_warning(self) -> None:
        module = _load_filter_module()

        self.assertTrue(
            module.should_suppress(
                "'+ptx85' is not a recognized feature for this target (ignoring feature)"
            )
        )


if __name__ == "__main__":
    unittest.main()
