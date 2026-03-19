import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


def _reload_evaluate():
    sys.modules.pop("evaluate", None)
    return importlib.import_module("evaluate")


class EvaluateHelpersTests(unittest.TestCase):
    def test_device_for_sahi_maps_int(self) -> None:
        ev = _reload_evaluate()
        self.assertEqual(ev.device_for_sahi(0), "cuda:0")
        self.assertEqual(ev.device_for_sahi(-1), "cpu")

    def test_device_for_sahi_maps_list(self) -> None:
        ev = _reload_evaluate()
        self.assertEqual(ev.device_for_sahi([1, 2]), "cuda:1")

    def test_load_dataset_config_from_yaml(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: grain\n",
                encoding="utf-8",
            )
            (root / "images" / "val").mkdir(parents=True)
            ds_root, cfg = ev.load_dataset_config_from_yaml(yaml_path)
            self.assertEqual(ds_root, root.resolve())
            self.assertEqual(cfg["val"], "images/val")


class EvaluateMainTests(unittest.TestCase):
    def test_parse_args_requires_mode_and_weights(self) -> None:
        ev = _reload_evaluate()
        with self.assertRaises(SystemExit):
            ev.parse_args([])

    def test_parse_args_sahi_requires_out_dir(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ev.parse_args(
                    [
                        "--mode",
                        "sahi",
                        "--weights",
                        "/w.pt",
                        "--data",
                        str(y),
                    ]
                )

    def test_main_routes_sahi(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y = root / "data.yaml"
            y.write_text("path: .\ntrain: t\nval: v\n", encoding="utf-8")
            out = root / "sahi_out"
            with patch.object(ev, "run_sahi") as mock_sahi:
                ev.main(
                    [
                        "--mode",
                        "sahi",
                        "--weights",
                        "/fake/weights.pt",
                        "--data",
                        str(y),
                        "--sahi-out-dir",
                        str(out),
                    ]
                )
            mock_sahi.assert_called_once()
            call_args, _ = mock_sahi.call_args
            self.assertEqual(call_args[0].mode, "sahi")
            self.assertEqual(call_args[1], y.resolve())

    @patch("ultralytics.YOLO")
    def test_run_val_forwards_kwargs(self, mock_yolo_cls: MagicMock) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            model = MagicMock()
            mock_yolo_cls.return_value = model
            args = SimpleNamespace(
                weights=str(Path(tmp) / "w.pt"),
                device="0",
                imgsz=640,
                batch=4,
                workers=2,
                plots=False,
                half=True,
                save_json=True,
                project=Path(tmp) / "proj",
                name="ev1",
            )
            Path(args.weights).write_bytes(b"")
            ev.run_val(args, y)
            model.val.assert_called_once()
            call_kw = model.val.call_args.kwargs
            self.assertEqual(call_kw["data"], str(y))
            self.assertEqual(call_kw["imgsz"], 640)
            self.assertEqual(call_kw["split"], "val")
            self.assertTrue(call_kw["save_json"])

    @patch("ultralytics.utils.benchmarks.benchmark")
    def test_run_benchmark_passes_format(self, mock_benchmark: MagicMock) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "d.yaml"
            y.write_text("path: .\nval: v\n", encoding="utf-8")
            w = Path(tmp) / "w.pt"
            w.write_bytes(b"")
            args = SimpleNamespace(
                weights=str(w),
                data=y,
                device="cpu",
                imgsz=320,
                half=False,
                verbose=True,
                format="onnx",
            )
            ev.run_benchmark(args, y)
            mock_benchmark.assert_called_once()
            kw = mock_benchmark.call_args.kwargs
            self.assertEqual(kw["format"], "onnx")
            self.assertEqual(kw["data"], str(y))


if __name__ == "__main__":
    unittest.main()
