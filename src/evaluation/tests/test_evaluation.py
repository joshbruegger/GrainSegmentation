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


class EvaluateValidationTests(unittest.TestCase):
    def test_validate_args_rejects_non_positive_patch_size(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = SimpleNamespace(
            num_inputs=1,
            image_suffixes=["_PPL"],
            patch_size=0,
            stride=64,
            batch_size=1,
            boundary_tolerance=2.0,
        )

        with self.assertRaisesRegex(ValueError, "patch_size and stride must be > 0"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_invalid_num_inputs(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = SimpleNamespace(
            num_inputs=3,
            image_suffixes=["_PPL", "_PPX1", "_PPX2"],
            patch_size=128,
            stride=64,
            batch_size=1,
            boundary_tolerance=2.0,
        )

        with self.assertRaisesRegex(ValueError, "num_inputs"):
            evaluate._validate_args(args)

    def test_validate_args_rejects_non_finite_boundary_tolerance(self) -> None:
        _install_evaluate_import_stubs()
        evaluate = _reload_module("evaluation.evaluate")

        args = SimpleNamespace(
            num_inputs=1,
            image_suffixes=["_PPL"],
            patch_size=128,
            stride=64,
            batch_size=1,
            boundary_tolerance=np.inf,
        )

        with self.assertRaisesRegex(
            ValueError, "boundary_tolerance must be finite and >= 0"
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


if __name__ == "__main__":
    unittest.main()
