import contextlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _install_tensorflow_stub() -> None:
    tf_module = types.ModuleType("tensorflow")
    tf_module.Tensor = object
    tf_module.keras = SimpleNamespace(
        Model=object,
        models=SimpleNamespace(load_model=lambda *args, **kwargs: None),
    )
    sys.modules["tensorflow"] = tf_module


def _install_evaluate_import_stubs() -> None:
    _install_tensorflow_stub()

    training_pkg = types.ModuleType("training")
    data_module = types.ModuleType("training.data")
    model_module = types.ModuleType("training.model")

    data_module.list_samples = lambda *args, **kwargs: []
    data_module._load_rgb_image = lambda path: np.zeros((2, 2, 3), dtype=np.float32)
    data_module._load_raster_mask = lambda path: np.zeros((2, 2), dtype=np.int32)
    model_module.weighted_crossentropy = lambda y_true, y_pred: 0.0

    training_pkg.data = data_module
    training_pkg.model = model_module

    sys.modules["training"] = training_pkg
    sys.modules["training.data"] = data_module
    sys.modules["training.model"] = model_module


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class MetricsTests(unittest.TestCase):
    def test_compute_aji_penalizes_merged_predictions(self) -> None:
        metrics = _reload_module("evaluation.metrics")

        true_instances = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
            ],
            dtype=np.int32,
        )
        pred_instances = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        self.assertAlmostEqual(
            metrics.compute_aji(true_instances, pred_instances),
            1.0 / 3.0,
        )


class InferenceTests(unittest.TestCase):
    def test_predict_full_image_uses_training_style_edge_starts(self) -> None:
        _install_tensorflow_stub()
        inference = _reload_module("evaluation.inference")

        recorded_starts = []

        class DummyModel:
            output_shape = (None, None, None, 2)

            def predict(self, batch, verbose=0):
                recorded_starts.extend(int(value) for value in batch[:, 0, 0, 0])
                return np.zeros(
                    (batch.shape[0], batch.shape[1], batch.shape[2], 2),
                    dtype=np.float32,
                )

        image = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
        inference.predict_full_image(
            model=DummyModel(),
            inputs=(image,),
            patch_size=4,
            stride=3,
            batch_size=8,
        )

        self.assertEqual(recorded_starts, [0, 1, 5, 6])


def _eval_validate_args_ns(**kwargs):
    defaults = {
        "num_inputs": 1,
        "image_suffixes": ["_PPL"],
        "patch_size": 128,
        "stride": 64,
        "batch_size": 1,
        "boundary_tolerance": 2.0,
        "coco_mask_ap": False,
        "instance_method": "cc",
        "watershed_min_distance": 1,
        "watershed_boundary_dilate_iter": 0,
        "watershed_connectivity": 1,
        "watershed_min_area_px": 0,
        "watershed_exclude_border": False,
        "watershed_ridge_level": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class EvaluateValidationTests(unittest.TestCase):
    def test_validate_args_rejects_non_positive_patch_size(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(patch_size=0)

        with self.assertRaisesRegex(ValueError, "patch_size and stride must be > 0"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_invalid_num_inputs(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(
            num_inputs=3,
            image_suffixes=["_PPL", "_PPX1", "_PPX2"],
        )

        with self.assertRaisesRegex(ValueError, "num_inputs"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_non_finite_boundary_tolerance(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(boundary_tolerance=np.inf)

        with self.assertRaisesRegex(
            ValueError, "boundary_tolerance must be finite and >= 0"
        ):
            evaluate._validate_args(args)

    def test_validate_args_rejects_negative_watershed_min_area_px(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(watershed_min_area_px=-1)

        with self.assertRaisesRegex(ValueError, "watershed_min_area_px must be >= 0"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_non_finite_watershed_ridge_level(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = _eval_validate_args_ns(watershed_ridge_level=float("nan"))

        with self.assertRaisesRegex(
            ValueError, "watershed_ridge_level must be finite when set"
        ):
            evaluate._validate_args(args)

    def test_validate_sample_data_rejects_mask_shape_mismatch(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        images = [np.zeros((2, 2, 3), dtype=np.float32)]
        mask = np.zeros((3, 3), dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "does not match image shape"):
            evaluate._validate_sample_data(images, mask, "mask.png")


class PlotResultsCliTests(unittest.TestCase):
    def test_main_exits_on_mismatched_quantitative_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--json-files",
            "a.json",
            "b.json",
            "--labels",
            "only-one",
            "--output-plot",
            "out.png",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                plot_results.main()

    def test_main_exits_on_incomplete_overlay_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--image-path",
            "image.png",
            "--gt-path",
            "gt.png",
            "--output-overlay",
            "overlay.png",
        ]

        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                plot_results.main()

    def test_main_accepts_complete_overlay_only_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        argv = [
            "plot_results.py",
            "--image-path",
            "image.png",
            "--gt-path",
            "gt.png",
            "--pred-paths",
            "pred.png",
            "--labels",
            "baseline",
            "--output-overlay",
            "overlay.png",
        ]

        with patch.object(sys, "argv", argv):
            with patch.object(plot_results, "generate_qualitative_overlay") as overlay:
                plot_results.main()

        overlay.assert_called_once_with(
            "image.png",
            "gt.png",
            ["pred.png"],
            ["baseline"],
            "overlay.png",
        )

    def test_generate_quantitative_plot_omits_error_bars_for_single_sample(
        self,
    ) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            json_path.write_text(
                json.dumps(
                    {
                        "heldout_section": {
                            "iou_class_1": 0.61,
                            "iou_class_2": 0.42,
                            "boundary_f1": 0.57,
                            "aji": 0.38,
                        }
                    }
                )
            )
            output_path = Path(tmpdir) / "plot.png"

            fig = object()
            ax = SimpleNamespace(
                bar=Mock(),
                set_ylabel=Mock(),
                set_title=Mock(),
                set_xticks=Mock(),
                set_xticklabels=Mock(),
                legend=Mock(),
                grid=Mock(),
            )

            with patch.object(plot_results.plt, "subplots", return_value=(fig, ax)):
                with patch.object(plot_results.plt, "tight_layout"):
                    with patch.object(plot_results.plt, "savefig"):
                        plot_results.generate_quantitative_plot(
                            [str(json_path)], ["Baseline"], str(output_path)
                        )

            self.assertNotIn("yerr", ax.bar.call_args.kwargs)
            self.assertIn("descriptive", ax.set_title.call_args.args[0].lower())

    def test_generate_qualitative_overlay_disables_pillow_pixel_guard(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        def guarded_open(path):
            if plot_results.Image.MAX_IMAGE_PIXELS is not None:
                raise plot_results.Image.DecompressionBombError("pixel guard enabled")

            if path == "image.png":
                image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
            else:
                image = Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L")

            return contextlib.closing(image)

        axes = [
            SimpleNamespace(imshow=Mock(), set_title=Mock(), axis=Mock())
            for _ in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"

            with patch.object(plot_results.Image, "open", side_effect=guarded_open):
                plot_results.generate_qualitative_overlay(
                    "image.png",
                    "gt.png",
                    ["pred.png"],
                    ["Baseline"],
                    str(output_path),
                )

            self.assertTrue((Path(tmpdir) / "overlay_ground_truth.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_Baseline.png").exists())

    def test_blend_overlay_uses_red_tint_for_all_foreground_classes(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        image = np.zeros((1, 3, 3), dtype=np.float32)
        mask = np.array([[0, 1, 2]], dtype=np.uint8)

        overlay = plot_results.blend_overlay(image, mask)

        np.testing.assert_allclose(overlay[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(overlay[0, 1], [0.4, 0.0, 0.0])
        np.testing.assert_allclose(overlay[0, 2], [0.4, 0.0, 0.0])

    def test_generate_qualitative_overlay_writes_ground_truth_and_one_file_per_model(
        self,
    ) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
        gt = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        pred_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        pred_b = np.array([[0, 0], [2, 1]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            gt_path = Path(tmpdir) / "gt.png"
            pred_a_path = Path(tmpdir) / "pred_a.png"
            pred_b_path = Path(tmpdir) / "pred_b.png"
            output_path = Path(tmpdir) / "overlay.png"

            Image.fromarray(rgb, mode="RGB").save(image_path)
            Image.fromarray(gt, mode="L").save(gt_path)
            Image.fromarray(pred_a, mode="L").save(pred_a_path)
            Image.fromarray(pred_b, mode="L").save(pred_b_path)
            output_path.write_text("stale montage", encoding="utf-8")

            plot_results.generate_qualitative_overlay(
                str(image_path),
                str(gt_path),
                [str(pred_a_path), str(pred_b_path)],
                ["ModelA", "ModelB"],
                str(output_path),
            )

            self.assertFalse(output_path.exists())
            self.assertTrue((Path(tmpdir) / "overlay_ground_truth.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_ModelA.png").exists())
            self.assertTrue((Path(tmpdir) / "overlay_ModelB.png").exists())

    def test_resize_overlay_arrays_downscales_large_inputs(self) -> None:
        plot_results = _reload_module("evaluation.plot_results")

        rgb_img = np.zeros((5000, 2500, 3), dtype=np.float32)
        gt_mask = np.zeros((5000, 2500), dtype=np.uint8)
        pred_mask = np.zeros((5000, 2500), dtype=np.uint8)

        resized_image, resized_gt, resized_preds = plot_results._resize_overlay_arrays(
            rgb_img, gt_mask, [pred_mask], max_dim=2048
        )

        self.assertEqual(resized_image.shape[:2], resized_gt.shape)
        self.assertEqual(resized_preds[0].shape, resized_gt.shape)
        self.assertLessEqual(max(resized_image.shape[:2]), 2048)
        self.assertGreater(min(resized_image.shape[:2]), 0)


class EvaluateSampleLoadingTests(unittest.TestCase):
    def test_validate_sample_data_rejects_out_of_range_labels(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        images = [np.zeros((2, 2, 3), dtype=np.float32)]
        mask = np.array([[0, 1], [2, 3]], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "Mask values must be in \\[0, 2\\]"):
            evaluate._validate_sample_data(images, mask, "mask.png")


class EvaluateMainTests(unittest.TestCase):
    def test_main_uses_descriptive_single_sample_output_contract(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            pred_dir = Path(tmpdir) / "preds"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--save-predictions-dir",
                str(pred_dir),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
            ]

            stdout = io.StringIO()
            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "compute_semantic_metrics",
                                    return_value={
                                        "iou_class_0": 1.0,
                                        "dice_class_0": 1.0,
                                        "iou_class_1": 0.75,
                                        "dice_class_1": 0.86,
                                        "iou_class_2": 0.5,
                                        "dice_class_2": 0.67,
                                    },
                                ):
                                    with patch.object(
                                        evaluate,
                                        "compute_boundary_f1",
                                        return_value=0.8,
                                    ):
                                        with patch.object(
                                            evaluate,
                                            "compute_boundary_iou",
                                            return_value=0.6,
                                        ):
                                            with patch.object(
                                                evaluate,
                                                "compute_aji",
                                                return_value=0.7,
                                            ):
                                                with contextlib.redirect_stdout(stdout):
                                                    evaluate.main()

            saved = json.loads(output_json.read_text())
            self.assertIn("heldout_section", saved)
            self.assertNotIn("mean", saved)
            self.assertEqual(saved["heldout_section"]["boundary_f1"], 0.8)
            self.assertTrue((pred_dir / "heldout_section_pred.png").exists())
            self.assertIn("descriptive", stdout.getvalue().lower())

    def test_main_coco_mask_ap_adds_coco_fields(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        from evaluation.coco_mask_ap import InstanceAPSummary

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)

        fake_coco = InstanceAPSummary(
            0.1,
            0.2,
            0.3,
            -1.0,
            -1.0,
            -1.0,
            0.4,
            0.5,
            0.6,
            None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
                "--coco-mask-ap",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "compute_semantic_metrics",
                                    return_value={
                                        "iou_class_0": 1.0,
                                        "dice_class_0": 1.0,
                                        "iou_class_1": 0.75,
                                        "dice_class_1": 0.86,
                                        "iou_class_2": 0.5,
                                        "dice_class_2": 0.67,
                                    },
                                ):
                                    with patch.object(
                                        evaluate,
                                        "compute_boundary_f1",
                                        return_value=0.8,
                                    ):
                                        with patch.object(
                                            evaluate,
                                            "compute_boundary_iou",
                                            return_value=0.6,
                                        ):
                                            with patch.object(
                                                evaluate,
                                                "compute_aji",
                                                return_value=0.7,
                                            ):
                                                with patch.object(
                                                    evaluate,
                                                    "evaluate_mask_ap",
                                                    return_value=fake_coco,
                                                ):
                                                    evaluate.main()

            saved = json.loads(output_json.read_text())
            self.assertEqual(saved["heldout_section"]["AP"], 0.1)
            self.assertEqual(saved["heldout_section"]["AP50"], 0.2)

    def test_main_uses_selected_instance_method_for_aji_and_coco(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)
        cc_instances = np.array([[0, 1], [0, 1]], dtype=np.int32)
        ws_instances = np.array([[0, 1], [0, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
                "--coco-mask-ap",
                "--instance-method",
                "watershed",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "compute_semantic_metrics",
                                    return_value={
                                        "iou_class_0": 1.0,
                                        "dice_class_0": 1.0,
                                        "iou_class_1": 0.75,
                                        "dice_class_1": 0.86,
                                        "iou_class_2": 0.5,
                                        "dice_class_2": 0.67,
                                    },
                                ):
                                    with patch.object(
                                        evaluate,
                                        "compute_boundary_f1",
                                        return_value=0.8,
                                    ):
                                        with patch.object(
                                            evaluate,
                                            "compute_boundary_iou",
                                            return_value=0.6,
                                        ):
                                            with patch.object(
                                                evaluate,
                                                "get_instances",
                                                return_value=cc_instances,
                                            ):
                                                with patch.object(
                                                    evaluate,
                                                    "semantic_to_instance_label_map_watershed",
                                                    return_value=ws_instances,
                                                ) as ws_fn:
                                                    with patch.object(
                                                        evaluate,
                                                        "compute_aji",
                                                        return_value=0.7,
                                                    ) as compute_aji:
                                                        with patch.object(
                                                            evaluate,
                                                            "instance_label_map_to_coco_gt",
                                                            return_value=[],
                                                        ):
                                                            with patch.object(
                                                                evaluate,
                                                                "instance_label_map_to_coco_dt",
                                                                return_value=[],
                                                            ) as dt_builder:
                                                                with patch.object(
                                                                    evaluate,
                                                                    "evaluate_mask_ap",
                                                                ) as coco_eval:
                                                                    coco_eval.return_value.to_dict.return_value = {
                                                                        "AP": 0.1,
                                                                        "AP50": 0.2,
                                                                        "AP75": 0.3,
                                                                        "APs": -1.0,
                                                                        "APm": -1.0,
                                                                        "APl": -1.0,
                                                                        "AR1": 0.4,
                                                                        "AR10": 0.5,
                                                                        "AR100": 0.6,
                                                                    }
                                                                    evaluate.main()

            compute_aji.assert_called_once_with(cc_instances, ws_instances)
            dt_builder.assert_called_once_with(
                ws_instances, image_id=1, height=2, width=2
            )
            ws_fn.assert_called_once()
            ws_kw = ws_fn.call_args[1]
            self.assertEqual(ws_kw["min_distance"], 1)
            self.assertEqual(ws_kw["boundary_dilate_iter"], 0)
            self.assertEqual(ws_kw["watershed_connectivity"], 1)
            self.assertEqual(ws_kw["min_area_px"], 0)
            self.assertFalse(ws_kw["exclude_border"])
            self.assertIsNone(ws_kw["ridge_level"])

    def test_main_passes_extended_watershed_cli_to_label_map(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        sample = {"id": "heldout_section", "images": ["img.png"], "mask": "mask.png"}
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        pred = np.array([[0, 1], [2, 1]], dtype=np.int32)
        cc_instances = np.array([[0, 1], [0, 1]], dtype=np.int32)
        ws_instances = np.array([[0, 1], [0, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "metrics.json"
            argv = [
                "evaluate.py",
                "--model-path",
                "model.keras",
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--output-json",
                str(output_json),
                "--num-inputs",
                "1",
                "--image-suffixes",
                "_PPL",
                "--instance-method",
                "watershed",
                "--watershed-min-distance",
                "5",
                "--watershed-boundary-dilate-iter",
                "1",
                "--watershed-connectivity",
                "2",
                "--watershed-min-area-px",
                "10",
                "--watershed-exclude-border",
                "--watershed-ridge-level",
                "3.5",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(evaluate, "list_samples", return_value=[sample]):
                    with patch.object(
                        evaluate, "_load_rgb_image", return_value=np.zeros((2, 2, 3))
                    ):
                        with patch.object(
                            evaluate, "_load_raster_mask", return_value=mask
                        ):
                            with patch.object(
                                evaluate,
                                "predict_full_image",
                                return_value=(pred, np.zeros((2, 2, 3))),
                            ):
                                with patch.object(
                                    evaluate,
                                    "compute_semantic_metrics",
                                    return_value={
                                        "iou_class_0": 1.0,
                                        "dice_class_0": 1.0,
                                        "iou_class_1": 0.75,
                                        "dice_class_1": 0.86,
                                        "iou_class_2": 0.5,
                                        "dice_class_2": 0.67,
                                    },
                                ):
                                    with patch.object(
                                        evaluate,
                                        "compute_boundary_f1",
                                        return_value=0.8,
                                    ):
                                        with patch.object(
                                            evaluate,
                                            "compute_boundary_iou",
                                            return_value=0.6,
                                        ):
                                            with patch.object(
                                                evaluate,
                                                "get_instances",
                                                return_value=cc_instances,
                                            ):
                                                with patch.object(
                                                    evaluate,
                                                    "semantic_to_instance_label_map_watershed",
                                                    return_value=ws_instances,
                                                ) as ws_fn:
                                                    with patch.object(
                                                        evaluate,
                                                        "compute_aji",
                                                        return_value=0.7,
                                                    ):
                                                        evaluate.main()

            ws_fn.assert_called_once()
            ws_kw = ws_fn.call_args[1]
            self.assertEqual(ws_kw["min_distance"], 5)
            self.assertEqual(ws_kw["boundary_dilate_iter"], 1)
            self.assertEqual(ws_kw["watershed_connectivity"], 2)
            self.assertEqual(ws_kw["min_area_px"], 10)
            self.assertTrue(ws_kw["exclude_border"])
            self.assertEqual(ws_kw["ridge_level"], 3.5)


if __name__ == "__main__":
    unittest.main()
