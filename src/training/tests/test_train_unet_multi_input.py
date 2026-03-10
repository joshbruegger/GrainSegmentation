import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_TRAINING = Path(__file__).resolve().parents[1]
if str(REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(REPO_TRAINING))


def _install_train_stub(calls: list[dict]) -> None:
    train_module = types.ModuleType("train")

    def train_model(**kwargs):
        calls.append(kwargs)

    train_module.train_model = train_model
    sys.modules["train"] = train_module


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class TrainCliTests(unittest.TestCase):
    def test_parse_args_help_describes_epochs_as_tuned_run_cap(self) -> None:
        _install_train_stub([])
        module = _reload_module("train_unet_multi_input")

        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as exc_info, redirect_stdout(buffer):
            module.parse_args(["--help"])

        self.assertEqual(exc_info.exception.code, 0)
        self.assertIn("frozen-CV pass", buffer.getvalue())

    def test_parse_args_accepts_resume_flag(self) -> None:
        _install_train_stub([])
        module = _reload_module("train_unet_multi_input")

        args = module.parse_args(
            [
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-model",
                "model.keras",
                "--resume",
                "latest.keras",
            ]
        )

        self.assertEqual(args.resume, "latest.keras")

    def test_main_rejects_checkpoint_and_resume(self) -> None:
        _install_train_stub([])
        module = _reload_module("train_unet_multi_input")

        with self.assertRaisesRegex(
            ValueError, "Use only one of --checkpoint or --resume"
        ):
            module.main(
                [
                    "--image-dir",
                    "images",
                    "--mask-dir",
                    "masks",
                    "--output-model",
                    "model.keras",
                    "--checkpoint",
                    "start.keras",
                    "--resume",
                    "latest.keras",
                ]
            )

    def test_main_passes_resume_to_train_model(self) -> None:
        calls: list[dict] = []
        _install_train_stub(calls)
        module = _reload_module("train_unet_multi_input")

        module.main(
            [
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-model",
                "model.keras",
                "--resume",
                "latest.keras",
            ]
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["resume_path"], "latest.keras")


if __name__ == "__main__":
    unittest.main()
