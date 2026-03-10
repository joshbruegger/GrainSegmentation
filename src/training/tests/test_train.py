import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_TRAINING = Path(__file__).resolve().parents[1]
if str(REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(REPO_TRAINING))


class _FakeIterations:
    def __init__(self, value: int) -> None:
        self._value = value

    def numpy(self) -> int:
        return self._value


class _FakeOptimizer:
    def __init__(self, iterations: int) -> None:
        self.iterations = _FakeIterations(iterations)


class _FakeStrategy:
    num_replicas_in_sync = 1

    def scope(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeHyperparameters:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def get(self, key: str):
        return self.values[key]


class _FakeHistory:
    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history


class _FakeFoldModel:
    def __init__(self, val_losses: list[float]) -> None:
        self.val_losses = val_losses
        self.fit_calls: list[dict] = []

    def fit(self, train_dataset, **kwargs):
        self.fit_calls.append({"train_dataset": train_dataset, **kwargs})
        return _FakeHistory({"val_loss": list(self.val_losses)})


def _install_training_import_stubs() -> None:
    np_module = types.ModuleType("numpy")
    tf_module = types.ModuleType("tensorflow")
    tf_module.errors = SimpleNamespace(ResourceExhaustedError=RuntimeError)
    tf_module.config = SimpleNamespace(list_logical_devices=lambda *_args, **_kwargs: [])
    tf_module.distribute = SimpleNamespace(MirroredStrategy=lambda: _FakeStrategy())
    tf_module.keras = SimpleNamespace(
        mixed_precision=SimpleNamespace(set_global_policy=lambda policy: None),
        models=SimpleNamespace(load_model=lambda *args, **kwargs: None),
        callbacks=SimpleNamespace(
            EarlyStopping=object,
            ReduceLROnPlateau=object,
            ModelCheckpoint=object,
            TensorBoard=object,
        ),
        Model=object,
    )

    kt_module = types.ModuleType("keras_tuner")

    class _BayesianOptimization:
        def __init__(self, *args, **kwargs) -> None:
            self.init_args = args
            self.init_kwargs = kwargs

    kt_module.BayesianOptimization = _BayesianOptimization

    keras_module = types.ModuleType("keras")
    optimizers_module = types.ModuleType("keras.optimizers")

    class _Adam:
        def __init__(self, learning_rate) -> None:
            self.learning_rate = learning_rate

    optimizers_module.Adam = _Adam
    keras_module.optimizers = optimizers_module

    data_module = types.ModuleType("data")
    data_module.build_dataset = lambda *args, **kwargs: []
    data_module.list_samples = lambda *args, **kwargs: []
    data_module.create_spatial_cv_folds = lambda *args, **kwargs: []

    model_module = types.ModuleType("model")
    model_module.build_unet = lambda *args, **kwargs: "fresh-model"
    model_module.initialize_from_checkpoint = lambda *args, **kwargs: "checkpoint-model"
    model_module.weighted_crossentropy = object()

    sys.modules["numpy"] = np_module
    sys.modules["tensorflow"] = tf_module
    sys.modules["keras_tuner"] = kt_module
    sys.modules["keras"] = keras_module
    sys.modules["keras.optimizers"] = optimizers_module
    sys.modules["data"] = data_module
    sys.modules["model"] = model_module


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


class TrainHelperTests(unittest.TestCase):
    def test_print_training_image_paths_groups_region_samples_by_source_image(
        self,
    ) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        samples = [
            {
                "id": "tile_a",
                "images": ["a_PPL.tif", "a_PPX1.tif"],
                "region": (0, 10, 0, 10),
            },
            {
                "id": "tile_a",
                "images": ["a_PPL.tif", "a_PPX1.tif"],
                "region": (10, 20, 0, 10),
            },
            {
                "id": "tile_b",
                "images": ["b_PPL.tif", "b_PPX1.tif"],
                "region": (0, 10, 0, 10),
            },
        ]

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train.print_training_image_paths(samples, "Training image order")

        output = buffer.getvalue()
        self.assertEqual(output.count("a_PPL.tif"), 1)
        self.assertEqual(output.count("a_PPX1.tif"), 1)
        self.assertIn("region_count : 2", output)
        self.assertIn("unique_source_samples : 2", output)

    def test_build_model_for_tuning_ignores_resume_final_checkpoint(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        with (
            patch.object(train, "build_unet", return_value="fresh-model") as build_unet,
            patch.object(
                train.tf.keras.models, "load_model", return_value="resumed-model"
            ) as load_model,
        ):
            model = train.build_model_for_tuning(
                checkpoint_path=None,
                resume_path="latest.keras",
                patch_size=256,
                num_inputs=2,
                hp="hp",
            )

        self.assertEqual(model, "fresh-model")
        build_unet.assert_called_once_with(256, num_inputs=2, hp="hp")
        load_model.assert_not_called()

    def test_build_final_model_loads_resume_checkpoint(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        with (
            patch.object(
                train.tf.keras.models, "load_model", return_value="resumed-model"
            ) as load_model,
            patch.object(
                train, "initialize_from_checkpoint", return_value="checkpoint-model"
            ) as init_from_checkpoint,
        ):
            model = train.build_final_model(
                checkpoint_path=None,
                resume_path="latest.keras",
                patch_size=256,
                num_inputs=2,
                hp="hp",
            )

        self.assertEqual(model, "resumed-model")
        load_model.assert_called_once()
        init_from_checkpoint.assert_not_called()

    def test_infer_initial_epoch_uses_optimizer_iterations(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        model = SimpleNamespace(optimizer=_FakeOptimizer(iterations=23))
        dataset = [object(), object(), object(), object(), object()]

        initial_epoch = train.infer_initial_epoch(model, dataset)

        self.assertEqual(initial_epoch, 4)

    def test_create_tuner_uses_persistent_state(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        tuner = train.create_tuner(
            hypermodel="hypermodel",
            max_trials=7,
            tuning_dir="/tmp/tuning",
            run_name="stable_run",
            num_inputs=7,
        )

        self.assertEqual(
            tuner.init_kwargs["directory"], "/tmp/tuning/tuning_stable_run_7in"
        )
        self.assertEqual(tuner.init_kwargs["project_name"], "unet_tuning")
        self.assertFalse(tuner.init_kwargs["overwrite"])

    def test_select_best_epoch_from_history_uses_val_loss_argmin(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        best_epoch, best_val_loss = train.select_best_epoch_from_history(
            _FakeHistory({"val_loss": [1.4, 1.1, 1.3]}),
            monitor="val_loss",
        )

        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_val_loss, 1.1)

    def test_summarize_cv_epoch_selection_uses_update_space_and_epoch_cap(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        summary = train.summarize_cv_epoch_selection(
            fold_summaries=[
                {"fold_index": 1, "best_epoch": 6, "train_steps_per_epoch": 4},
                {"fold_index": 2, "best_epoch": 10, "train_steps_per_epoch": 4},
                {"fold_index": 3, "best_epoch": 20, "train_steps_per_epoch": 4},
            ],
            full_steps_per_epoch=8,
            epoch_cap=4,
        )

        self.assertEqual(summary["aggregated_updates"], 40)
        self.assertEqual(summary["unbounded_final_epochs"], 5)
        self.assertEqual(summary["selected_final_epochs"], 4)

    def test_estimate_final_epochs_from_cv_ignores_resume_checkpoint(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        fold_model = _FakeFoldModel([1.3, 1.1, 1.2])

        with (
            patch.object(
                train, "build_dataset", side_effect=[[object(), object()], [object()]]
            ),
            patch.object(
                train,
                "build_model_for_tuning",
                side_effect=[fold_model],
            ) as build_model_for_tuning,
            patch.object(
                train,
                "compile_model_for_training",
                side_effect=lambda model, learning_rate: model,
            ),
            patch.object(
                train.tf.keras.callbacks,
                "EarlyStopping",
                side_effect=lambda **kwargs: ("early-stopping", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ReduceLROnPlateau",
                side_effect=lambda **kwargs: ("reduce-lr", kwargs),
            ),
        ):
            summary = train.estimate_final_epochs_from_cv(
                cv_folds=[(["train"], ["val"])],
                checkpoint_path="start.keras",
                patch_size=256,
                stride=128,
                num_inputs=2,
                best_batch_size=4,
                best_hp=best_hp,
                max_epochs=12,
                full_steps_per_epoch=2,
            )

        self.assertEqual(summary["selected_final_epochs"], 2)
        self.assertFalse(summary["used_fallback"])
        self.assertEqual(
            fold_model.fit_calls[0]["callbacks"][1][0],
            "reduce-lr",
        )
        build_model_for_tuning.assert_called_once_with(
            checkpoint_path="start.keras",
            resume_path=None,
            patch_size=256,
            num_inputs=2,
            hp=best_hp,
        )

    def test_estimate_final_epochs_from_cv_returns_fallback_on_oom(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        oom_model = SimpleNamespace(
            fit=Mock(side_effect=train.tf.errors.ResourceExhaustedError("oom"))
        )

        with (
            patch.object(
                train, "build_dataset", side_effect=[[object(), object()], [object()]]
            ),
            patch.object(
                train, "build_model_for_tuning", return_value=oom_model
            ) as build_model_for_tuning,
            patch.object(
                train,
                "compile_model_for_training",
                side_effect=lambda model, learning_rate: model,
            ),
            patch.object(
                train.tf.keras.callbacks,
                "EarlyStopping",
                side_effect=lambda **kwargs: ("early-stopping", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ReduceLROnPlateau",
                side_effect=lambda **kwargs: ("reduce-lr", kwargs),
            ),
        ):
            summary = train.estimate_final_epochs_from_cv(
                cv_folds=[(["train"], ["val"])],
                checkpoint_path="start.keras",
                patch_size=256,
                stride=128,
                num_inputs=2,
                best_batch_size=4,
                best_hp=best_hp,
                max_epochs=12,
                full_steps_per_epoch=2,
            )

        self.assertTrue(summary["used_fallback"])
        self.assertEqual(summary["selected_final_epochs"], 12)
        self.assertIn("OOM", summary["fallback_reason"])
        build_model_for_tuning.assert_called_once()

    def test_train_model_uses_selected_cv_epochs_for_tuned_final_training(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)

        with (
            patch.object(train.tf.keras.callbacks, "TensorBoard", side_effect=lambda **kwargs: ("tensorboard", kwargs)),
            patch.object(train.tf.keras.callbacks, "ReduceLROnPlateau", side_effect=lambda **kwargs: ("reduce-lr", kwargs)),
            patch.object(train.tf.keras.callbacks, "ModelCheckpoint", side_effect=lambda **kwargs: ("checkpoint", kwargs)),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_cv_folds",
                return_value=[(["train-a"], ["val-a"]), (["train-b"], ["val-b"])],
            ),
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner),
            patch.object(train, "build_dataset", return_value=[object(), object(), object()]),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
            patch.object(
                train,
                "estimate_final_epochs_from_cv",
                return_value={
                    "selected_final_epochs": 7,
                    "aggregated_updates": 21,
                    "unbounded_final_epochs": 7,
                    "used_fallback": False,
                    "fold_summaries": [],
                },
            ) as estimate_final_epochs_from_cv,
        ):
            train.train_model(
                image_dir="images",
                mask_dir="masks",
                checkpoint_path="start.keras",
                resume_path="resume.keras",
                output_model_path="model.keras",
                patch_size=256,
                stride=128,
                tune_epochs=20,
                final_epochs=100,
                image_suffixes=["_PPL"],
                mask_ext=".tif",
                mask_stem_suffix="_labels",
                split_tile_size=256,
                split_coverage_bins=8,
                num_inputs=1,
                run_name="epoch-selection",
                tuning_dir="/tmp/tuning",
                n_splits=2,
                random_state=42,
                use_mixed_precision=False,
                max_trials=3,
                skip_tuning=False,
            )

        self.assertEqual(
            estimate_final_epochs_from_cv.call_args.kwargs["max_epochs"], 100
        )
        final_model.fit.assert_called_once()
        self.assertEqual(final_model.fit.call_args.kwargs["epochs"], 7)

    def test_train_model_skips_epoch_selection_when_resuming_without_checkpoint(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)

        with (
            patch.object(train.tf.keras.callbacks, "TensorBoard", side_effect=lambda **kwargs: ("tensorboard", kwargs)),
            patch.object(train.tf.keras.callbacks, "ReduceLROnPlateau", side_effect=lambda **kwargs: ("reduce-lr", kwargs)),
            patch.object(train.tf.keras.callbacks, "ModelCheckpoint", side_effect=lambda **kwargs: ("checkpoint", kwargs)),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_cv_folds",
                return_value=[(["train-a"], ["val-a"]), (["train-b"], ["val-b"])],
            ),
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner),
            patch.object(train, "build_dataset", return_value=[object(), object(), object()]),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
            patch.object(train, "infer_initial_epoch", return_value=3),
            patch.object(train, "estimate_final_epochs_from_cv") as estimate_final_epochs_from_cv,
        ):
            train.train_model(
                image_dir="images",
                mask_dir="masks",
                checkpoint_path=None,
                resume_path="resume.keras",
                output_model_path="model.keras",
                patch_size=256,
                stride=128,
                tune_epochs=20,
                final_epochs=100,
                image_suffixes=["_PPL"],
                mask_ext=".tif",
                mask_stem_suffix="_labels",
                split_tile_size=256,
                split_coverage_bins=8,
                num_inputs=1,
                run_name="epoch-selection",
                tuning_dir="/tmp/tuning",
                n_splits=2,
                random_state=42,
                use_mixed_precision=False,
                max_trials=3,
                skip_tuning=False,
            )

        estimate_final_epochs_from_cv.assert_not_called()
        final_model.fit.assert_called_once()
        self.assertEqual(final_model.fit.call_args.kwargs["epochs"], 100)

    def test_train_model_skips_fit_when_resume_already_meets_selected_epochs(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)

        with (
            patch.object(train.tf.keras.callbacks, "TensorBoard", side_effect=lambda **kwargs: ("tensorboard", kwargs)),
            patch.object(train.tf.keras.callbacks, "ReduceLROnPlateau", side_effect=lambda **kwargs: ("reduce-lr", kwargs)),
            patch.object(train.tf.keras.callbacks, "ModelCheckpoint", side_effect=lambda **kwargs: ("checkpoint", kwargs)),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_cv_folds",
                return_value=[(["train-a"], ["val-a"]), (["train-b"], ["val-b"])],
            ),
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner),
            patch.object(train, "build_dataset", return_value=[object(), object(), object()]),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
            patch.object(train, "infer_initial_epoch", return_value=7),
            patch.object(
                train,
                "estimate_final_epochs_from_cv",
                return_value={
                    "selected_final_epochs": 7,
                    "aggregated_updates": 21,
                    "unbounded_final_epochs": 7,
                    "used_fallback": False,
                    "fold_summaries": [],
                },
            ),
        ):
            train.train_model(
                image_dir="images",
                mask_dir="masks",
                checkpoint_path="start.keras",
                resume_path="resume.keras",
                output_model_path="model.keras",
                patch_size=256,
                stride=128,
                tune_epochs=20,
                final_epochs=100,
                image_suffixes=["_PPL"],
                mask_ext=".tif",
                mask_stem_suffix="_labels",
                split_tile_size=256,
                split_coverage_bins=8,
                num_inputs=1,
                run_name="epoch-selection",
                tuning_dir="/tmp/tuning",
                n_splits=2,
                random_state=42,
                use_mixed_precision=False,
                max_trials=3,
                skip_tuning=False,
            )

        final_model.fit.assert_not_called()
        final_model.save.assert_called_once_with("model.keras")


if __name__ == "__main__":
    unittest.main()
