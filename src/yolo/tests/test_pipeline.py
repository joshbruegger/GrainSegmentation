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
        type(self).instances.append(self)

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
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
                model_source="yolo26x-seg.pt",
                resume_path=None,
                epochs=100,
                imgsz=1024,
                batch=-1,
                workers=16,
                device=[0, 1],
                cache="disk",
                amp=True,
                plots=True,
                exist_ok=True,
                yolo_factory=_FakeYOLO,
            )

        self.assertEqual(result["model"], "yolo26x-seg.pt")
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
                model_source="yolo26x-seg.pt",
                resume_path=resume_path,
                epochs=100,
                imgsz=1024,
                batch=-1,
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


if __name__ == "__main__":
    unittest.main()
