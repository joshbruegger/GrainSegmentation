import importlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


class _FakeYOLO:
    instances: list["_FakeYOLO"] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self.train_calls: list[dict] = []
        self.tune_calls: list[dict] = []
        type(self).instances.append(self)

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return {"model": self.model, "kwargs": kwargs}

    def tune(self, **kwargs):
        self.tune_calls.append(kwargs)
        return {"model": self.model, "kwargs": kwargs}


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeYOLO.instances.clear()

    def test_resolve_variant_paths_uses_scratch_dataset_root(self) -> None:
        pipeline = _reload_module("pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_root = Path(tmpdir)
            resolved = pipeline.resolve_variant_paths(
                variant_name="PPL+PPXblend",
                scratch_root=scratch_root,
            )

        self.assertEqual(
            resolved.data_yaml,
            scratch_root
            / "GrainSeg"
            / "dataset"
            / "MWD-1#121"
            / "yolo"
            / "PPL+PPXblend"
            / "PPL_PPXblend.yaml",
        )
        self.assertEqual(resolved.channels, 6)

    def test_default_resume_checkpoint_uses_project_name_weights_last_pt(self) -> None:
        pipeline = _reload_module("pipeline")

        checkpoint = pipeline.default_resume_checkpoint(
            project_dir=Path("/scratch/run-root"),
            run_name="PPL",
        )

        self.assertEqual(
            checkpoint,
            Path("/scratch/run-root") / "PPL" / "weights" / "last.pt",
        )

    def test_train_model_uses_weights_for_fresh_runs(self) -> None:
        pipeline = _reload_module("pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_yaml = Path(tmpdir) / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
            )

            result = pipeline.train_model(
                data_yaml=data_yaml,
                run_name="PPL",
                project_dir=Path(tmpdir) / "runs",
                model_source="yolo26l-seg.pt",
                resume_path=None,
                epochs=100,
                imgsz=1024,
                batch=-1,
                lr0=0.001,
                dropout=0.05,
                workers=16,
                device=[0, 1],
                cache="disk",
                amp=True,
                plots=True,
                exist_ok=True,
                yolo_factory=_FakeYOLO,
            )

        self.assertEqual(result["model"], "yolo26l-seg.pt")
        self.assertEqual(len(_FakeYOLO.instances), 1)
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["data"], str(data_yaml))
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["name"], "PPL")
        self.assertEqual(
            _FakeYOLO.instances[0].train_calls[0]["project"], str(Path(tmpdir) / "runs")
        )
        self.assertNotIn("resume", _FakeYOLO.instances[0].train_calls[0])

    def test_train_model_uses_resume_checkpoint_when_requested(self) -> None:
        pipeline = _reload_module("pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_yaml = Path(tmpdir) / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
            )
            resume_path = Path(tmpdir) / "runs" / "PPL" / "weights" / "last.pt"
            resume_path.parent.mkdir(parents=True)
            resume_path.write_text("checkpoint", encoding="utf-8")

            result = pipeline.train_model(
                data_yaml=data_yaml,
                run_name="PPL",
                project_dir=Path(tmpdir) / "runs",
                model_source="yolo26l-seg.pt",
                resume_path=resume_path,
                epochs=100,
                imgsz=1024,
                batch=-1,
                lr0=0.001,
                dropout=0.05,
                workers=16,
                device=[0, 1],
                cache="disk",
                amp=True,
                plots=True,
                exist_ok=True,
                yolo_factory=_FakeYOLO,
            )

        self.assertEqual(result["model"], str(resume_path))
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["resume"], True)
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["data"], str(data_yaml))
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["device"], [0, 1])
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["imgsz"], 1024)
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["batch"], -1)
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["workers"], 16)
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["cache"], "disk")
        self.assertEqual(_FakeYOLO.instances[0].train_calls[0]["plots"], True)

    def test_tune_model_uses_builtin_tuner_with_custom_search_space(self) -> None:
        pipeline = _reload_module("pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_yaml = Path(tmpdir) / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
            )

            result = pipeline.tune_model(
                data_yaml=data_yaml,
                run_name="PPL-tune",
                project_dir=Path(tmpdir) / "runs",
                model_source="yolo26l-seg.pt",
                epochs=30,
                iterations=300,
                imgsz=1024,
                batch=-1,
                lr0=1.5e-3,
                dropout=0.35,
                workers=16,
                device=[0, 1],
                cache="disk",
                amp=True,
                resume=False,
                exist_ok=True,
                yolo_factory=_FakeYOLO,
            )

        self.assertEqual(result["model"], "yolo26l-seg.pt")
        self.assertEqual(len(_FakeYOLO.instances), 1)
        tune_call = _FakeYOLO.instances[0].tune_calls[0]
        self.assertEqual(tune_call["data"], str(data_yaml))
        self.assertEqual(tune_call["epochs"], 30)
        self.assertEqual(tune_call["iterations"], 300)
        self.assertEqual(tune_call["device"], "0,1")
        self.assertEqual(tune_call["project"], str(Path(tmpdir) / "runs"))
        self.assertEqual(tune_call["name"], "PPL-tune")
        self.assertEqual(tune_call["space"]["lr0"], (6e-4, 2.5e-3))
        self.assertEqual(tune_call["space"]["dropout"], (0.1, 0.6))
        self.assertEqual(tune_call["lr0"], 1.5e-3)
        self.assertEqual(tune_call["dropout"], 0.35)
        self.assertNotIn("resume", tune_call)
        for disallowed in ("plots", "save", "val"):
            self.assertNotIn(disallowed, tune_call)

    def test_tune_model_passes_resume_flag_when_requested(self) -> None:
        pipeline = _reload_module("pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_yaml = Path(tmpdir) / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
            )

            pipeline.tune_model(
                data_yaml=data_yaml,
                run_name="PPL-tune",
                project_dir=Path(tmpdir) / "runs",
                model_source="yolo26l-seg.pt",
                epochs=30,
                iterations=300,
                imgsz=1024,
                batch=-1,
                lr0=1.5e-3,
                dropout=0.35,
                workers=16,
                device=0,
                cache="disk",
                amp=True,
                resume=True,
                exist_ok=True,
                yolo_factory=_FakeYOLO,
            )

        self.assertTrue(_FakeYOLO.instances[0].tune_calls[0]["resume"])


if __name__ == "__main__":
    unittest.main()
