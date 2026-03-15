import importlib
import sys
import unittest
from pathlib import Path


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class VariantConfigTests(unittest.TestCase):
    def test_get_variant_config_returns_expected_multichannel_variant(self) -> None:
        config = _reload_module("config")

        variant = config.get_variant_config("PPL+AllPPX")

        self.assertEqual(variant.name, "PPL+AllPPX")
        self.assertEqual(variant.dataset_subdir, "PPL+AllPPX")
        self.assertEqual(variant.yaml_name, "PPL+AllPPX.yaml")
        self.assertEqual(variant.channels, 32)
        self.assertEqual(variant.slurm_mem, "950G")

    def test_get_variant_config_rejects_unknown_variant(self) -> None:
        config = _reload_module("config")

        with self.assertRaisesRegex(ValueError, "Unknown YOLO variant"):
            config.get_variant_config("unknown")


if __name__ == "__main__":
    unittest.main()
