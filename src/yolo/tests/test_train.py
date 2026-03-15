import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


def _install_pipeline_stub(calls: list[dict]) -> None:
    pipeline_module = types.ModuleType("pipeline")

    def train_model(**kwargs):
        calls.append(kwargs)
        return kwargs

    def resolve_variant_paths(*, variant_name, scratch_root=None):
        return types.SimpleNamespace(
            variant_name=variant_name,
            data_yaml=Path("/scratch/fake") / variant_name / "dataset.yaml",
            channels=1,
        )

    def default_project_dir(*, scratch_root=None):
        return Path("/scratch/fake-runs")

    def default_resume_checkpoint(*, project_dir, run_name):
        return Path(project_dir) / run_name / "weights" / "last.pt"

    pipeline_module.train_model = train_model
    pipeline_module.resolve_variant_paths = resolve_variant_paths
    pipeline_module.default_project_dir = default_project_dir
    pipeline_module.default_resume_checkpoint = default_resume_checkpoint
    sys.modules["pipeline"] = pipeline_module


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class TrainCliTests(unittest.TestCase):
    def test_parse_args_help_mentions_yolo26_seg(self) -> None:
        _install_pipeline_stub([])
        module = _reload_module("train")

        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as exc_info, redirect_stdout(buffer):
            module.parse_args(["--help"])

        self.assertEqual(exc_info.exception.code, 0)
        self.assertIn("yolo26x-seg", buffer.getvalue())

    def test_parse_args_requires_variant_or_data(self) -> None:
        _install_pipeline_stub([])
        module = _reload_module("train")

        with self.assertRaises(SystemExit):
            module.parse_args([])

    def test_main_rejects_resume_and_resume_checkpoint_together(self) -> None:
        _install_pipeline_stub([])
        module = _reload_module("train")

        with self.assertRaisesRegex(
            ValueError, "Use only one of --resume or --resume-checkpoint"
        ):
            module.main(
                [
                    "--variant",
                    "PPL",
                    "--resume",
                    "--resume-checkpoint",
                    "/tmp/last.pt",
                ]
            )

    def test_main_rejects_epochs_override_when_resuming(self) -> None:
        _install_pipeline_stub([])
        module = _reload_module("train")

        with self.assertRaisesRegex(
            ValueError,
            "Ultralytics does not support overriding --epochs while resuming",
        ):
            module.main(
                [
                    "--variant",
                    "PPL",
                    "--resume",
                    "--epochs",
                    "200",
                ]
            )

    def test_main_rejects_amp_override_when_resuming(self) -> None:
        _install_pipeline_stub([])
        module = _reload_module("train")

        with self.assertRaisesRegex(
            ValueError, "Ultralytics does not support overriding --amp while resuming"
        ):
            module.main(
                [
                    "--variant",
                    "PPL",
                    "--resume",
                    "--no-amp",
                ]
            )

    def test_main_uses_variant_defaults_and_default_resume_checkpoint(self) -> None:
        calls: list[dict] = []
        _install_pipeline_stub(calls)
        module = _reload_module("train")

        module.main(["--variant", "PPL", "--resume"])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["run_name"], "PPL")
        self.assertEqual(
            calls[0]["data_yaml"], Path("/scratch/fake") / "PPL" / "dataset.yaml"
        )
        self.assertEqual(
            calls[0]["resume_path"],
            Path("/scratch/fake-runs") / "PPL" / "weights" / "last.pt",
        )

    def test_main_allows_explicit_data_override(self) -> None:
        calls: list[dict] = []
        _install_pipeline_stub(calls)
        module = _reload_module("train")

        module.main(
            [
                "--data",
                "/tmp/custom.yaml",
                "--name",
                "custom-run",
            ]
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["run_name"], "custom-run")
        self.assertEqual(calls[0]["data_yaml"], Path("/tmp/custom.yaml"))
        self.assertIsNone(calls[0]["resume_path"])


if __name__ == "__main__":
    unittest.main()
