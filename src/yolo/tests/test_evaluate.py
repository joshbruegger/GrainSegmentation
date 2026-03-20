import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

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
            self.assertEqual(call_kw["split"], "test")
            self.assertTrue(call_kw["save_json"])
            self.assertEqual(call_kw["name"], "ev1")

    @patch("ultralytics.YOLO")
    def test_run_val_forwards_name_without_project(
        self, mock_yolo_cls: MagicMock
    ) -> None:
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
                save_json=False,
                project=None,
                name="my_run",
            )
            Path(args.weights).write_bytes(b"")
            ev.run_val(args, y)
            model.val.assert_called_once()
            call_kw = model.val.call_args.kwargs
            self.assertEqual(call_kw["name"], "my_run")
            self.assertNotIn("project", call_kw)

    def test_load_sahi_pairs_manifest_relative_to_manifest_dir(self) -> None:
        ev = _reload_evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "data"
            sub.mkdir()
            (sub / "a.tif").write_bytes(b"")
            (sub / "a.gpkg").write_bytes(b"")
            manifest = sub / "manifest.json"
            manifest.write_text(
                json.dumps([{"test_tiff": "a.tif", "test_gpkg": "a.gpkg"}]),
                encoding="utf-8",
            )
            old = os.getcwd()
            try:
                os.chdir("/")
                args = SimpleNamespace(
                    manifest=manifest,
                    test_tiff=None,
                    test_gpkg=None,
                )
                pairs = ev._load_sahi_pairs(args)
            finally:
                os.chdir(old)
            self.assertEqual(pairs[0][0], (sub / "a.tif").resolve())
            self.assertEqual(pairs[0][1], (sub / "a.gpkg").resolve())

    @patch("sahi.predict.get_sliced_prediction")
    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_creates_parent_dirs_for_output_json(
        self,
        mock_from_pretrained: MagicMock,
        mock_sliced: MagicMock,
    ) -> None:
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        mock_sliced.return_value = MagicMock(object_prediction_list=[])
        fake_img = np.zeros((10, 10, 1), dtype=np.uint8)
        with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
            with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    tiff = root / "tile.tif"
                    gpkg = root / "tile.gpkg"
                    tiff.write_bytes(b"")
                    gpkg.write_bytes(b"")
                    out = root / "nested" / "deep" / "metrics.json"
                    args = SimpleNamespace(
                        weights=str(root / "w.pt"),
                        device="cpu",
                        conf=0.25,
                        slice_height=64,
                        slice_width=64,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        test_tiff=tiff,
                        test_gpkg=gpkg,
                        manifest=None,
                        sahi_out_dir=None,
                        output_json=out,
                    )
                    (root / "w.pt").write_bytes(b"")
                    ev.run_sahi(args)
                    self.assertTrue(out.is_file())
                    self.assertTrue(out.parent.is_dir())

    def test_aggregate_sahi_means_excludes_undefined(self) -> None:
        ev = _reload_evaluate()
        row_ok = {
            "AP": 0.5,
            "AP50": 0.6,
            "AP75": 0.7,
            "APs": 0.1,
            "APm": 0.2,
            "APl": 0.3,
            "AR1": 0.4,
            "AR10": 0.5,
            "AR100": 0.6,
        }
        row_empty_gt = {
            "AP": -1.0,
            "AP50": -1.0,
            "AP75": -1.0,
            "APs": -1.0,
            "APm": -1.0,
            "APl": -1.0,
            "AR1": -1.0,
            "AR10": -1.0,
            "AR100": -1.0,
        }
        means = ev.aggregate_sahi_means([row_ok, row_empty_gt])
        self.assertAlmostEqual(means["mean_AP"], 0.5)
        self.assertAlmostEqual(means["mean_AP50"], 0.6)

    def test_aggregate_sahi_means_single_image_all_undefined(self) -> None:
        ev = _reload_evaluate()
        row = {
            "AP": -1.0,
            "AP50": -1.0,
            "AP75": -1.0,
            "APs": -1.0,
            "APm": -1.0,
            "APl": -1.0,
            "AR1": -1.0,
            "AR10": -1.0,
            "AR100": -1.0,
        }
        means = ev.aggregate_sahi_means([row])
        self.assertIsNone(means["mean_AP"])
        self.assertIsNone(means["mean_AP50"])

    @patch("sahi.predict.get_sliced_prediction")
    @patch("sahi.AutoDetectionModel.from_pretrained")
    def test_run_sahi_output_json_uses_null_not_nan_for_undefined_means(
        self,
        mock_from_pretrained: MagicMock,
        mock_sliced: MagicMock,
    ) -> None:
        """Written metrics.json must be strict JSON: null for undefined mean_*, no NaN."""
        ev = _reload_evaluate()
        mock_from_pretrained.return_value = MagicMock()
        mock_sliced.return_value = MagicMock(object_prediction_list=[])
        fake_img = np.zeros((10, 10, 1), dtype=np.uint8)
        with patch.object(ev, "load_image_for_yolo", return_value=fake_img):
            with patch.object(ev, "load_polygons_from_gpkg", return_value=[]):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    tiff = root / "tile.tif"
                    gpkg = root / "tile.gpkg"
                    tiff.write_bytes(b"")
                    gpkg.write_bytes(b"")
                    out = root / "m.json"
                    args = SimpleNamespace(
                        weights=str(root / "w.pt"),
                        device="cpu",
                        conf=0.25,
                        slice_height=64,
                        slice_width=64,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        test_tiff=tiff,
                        test_gpkg=gpkg,
                        manifest=None,
                        sahi_out_dir=None,
                        output_json=out,
                    )
                    (root / "w.pt").write_bytes(b"")
                    ev.run_sahi(args)
                    text = out.read_text(encoding="utf-8")
                    self.assertNotIn("NaN", text)
                    self.assertNotIn("Infinity", text)
                    loaded = json.loads(text)
                    self.assertIsNone(loaded["mean_AP"])
                    self.assertIsNone(loaded["mean_AP50"])


if __name__ == "__main__":
    unittest.main()
