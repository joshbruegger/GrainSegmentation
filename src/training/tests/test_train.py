import importlib
import io
import sys
import tempfile
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


class _FakeTrialHyperparameters(_FakeHyperparameters):
    def Float(self, key: str, **_kwargs):
        return self.values[key]


class _FakeHistory:
    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history


class _FakeFoldModel:
    def __init__(self, val_losses: list[float]) -> None:
        self.val_losses = val_losses
        self.fit_calls: list[dict] = []
        self.compile_calls: list[dict] = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)

    def fit(self, train_dataset, **kwargs):
        self.fit_calls.append({"train_dataset": train_dataset, **kwargs})
        return _FakeHistory({"val_loss": list(self.val_losses)})


def _install_training_import_stubs() -> None:
    np_module = types.ModuleType("numpy")
    np_module.mean = lambda values: sum(values) / len(values)
    tf_module = types.ModuleType("tensorflow")
    tf_module.errors = SimpleNamespace(ResourceExhaustedError=RuntimeError)
    tf_module.config = SimpleNamespace(
        list_logical_devices=lambda *_args, **_kwargs: []
    )
    tf_module.distribute = SimpleNamespace(MirroredStrategy=lambda: _FakeStrategy())
    tf_module.keras = SimpleNamespace(
        mixed_precision=SimpleNamespace(set_global_policy=lambda policy: None),
        models=SimpleNamespace(load_model=lambda *args, **kwargs: None),
        callbacks=SimpleNamespace(
            EarlyStopping=lambda **kwargs: ("early-stopping", kwargs),
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
    data_module.create_spatial_holdout_split = lambda *args, **kwargs: ([], [])

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
            validation_fraction=0.2,
        )

        self.assertEqual(
            tuner.init_kwargs["directory"], "/tmp/tuning/tuning_stable_run_7in_val20"
        )
        self.assertEqual(tuner.init_kwargs["project_name"], "unet_tuning")
        self.assertFalse(tuner.init_kwargs["overwrite"])

    def test_has_saved_tuner_state_is_scoped_by_validation_fraction(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")

        with tempfile.TemporaryDirectory() as tmpdir:
            oracle_dir = Path(tmpdir) / "tuning_stable_run_7in_val20" / "unet_tuning"
            oracle_dir.mkdir(parents=True)
            (oracle_dir / "oracle.json").write_text("{}", encoding="utf-8")

            self.assertTrue(
                train.has_saved_tuner_state(
                    tmpdir,
                    "stable_run",
                    7,
                    validation_fraction=0.2,
                )
            )
            self.assertFalse(
                train.has_saved_tuner_state(
                    tmpdir,
                    "stable_run",
                    7,
                    validation_fraction=0.3,
                )
            )

    def test_build_fold_cache_path_tracks_fold_contents_and_geometry(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        samples = [
            {
                "id": "tile-a",
                "images": ["a_PPL.tif", "a_PPX1.tif"],
                "mask": "a_labels.tif",
                "region": (0, 256, 0, 256),
            }
        ]

        with (
            patch.dict(train.os.environ, {"TMPDIR": "/tmp/cache-tests"}, clear=True),
            patch.object(train.os, "getpid", return_value=4321),
        ):
            cache_path = train.build_fold_cache_path(
                samples,
                role="train",
                patch_size=256,
                stride=128,
                num_inputs=2,
            )
            same_cache_path = train.build_fold_cache_path(
                list(samples),
                role="train",
                patch_size=256,
                stride=128,
                num_inputs=2,
            )
            different_stride_cache = train.build_fold_cache_path(
                samples,
                role="train",
                patch_size=256,
                stride=256,
                num_inputs=2,
            )
            different_sample_cache = train.build_fold_cache_path(
                [{**samples[0], "mask": "b_labels.tif"}],
                role="train",
                patch_size=256,
                stride=128,
                num_inputs=2,
            )

        self.assertEqual(cache_path, same_cache_path)
        self.assertNotEqual(cache_path, different_stride_cache)
        self.assertNotEqual(cache_path, different_sample_cache)
        self.assertNotIn("trial", cache_path)
        self.assertIn("local-4321", cache_path)

    def test_cv_tuner_run_trial_reuses_holdout_caches_across_trials(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        tuner = train.CVTuner()
        tuner.hypermodel = SimpleNamespace(
            build=lambda _hp: _FakeFoldModel([1.5, 1.2, 1.3])
        )

        def build_dataset_side_effect(*_args, **kwargs):
            return [object(), object()] if kwargs["augment"] else [object()]

        with (
            patch.object(
                train, "build_dataset", side_effect=build_dataset_side_effect
            ) as build_dataset,
            patch.object(
                train.tf.keras.callbacks,
                "EarlyStopping",
                side_effect=lambda **kwargs: ("early-stopping", kwargs),
            ),
            patch.dict(
                train.os.environ,
                {"TMPDIR": "/tmp/cache-tests", "SLURM_JOB_ID": "12345"},
                clear=False,
            ),
        ):
            trial_one = SimpleNamespace(
                trial_id="001",
                hyperparameters=_FakeTrialHyperparameters(
                    {"dropout": 0.2, "learning_rate": 1e-3}
                ),
            )
            trial_two = SimpleNamespace(
                trial_id="002",
                hyperparameters=_FakeTrialHyperparameters(
                    {"dropout": 0.3, "learning_rate": 5e-4}
                ),
            )

            tuner.run_trial(
                trial_one,
                train_samples=["train-a"],
                val_samples=["val-a"],
                patch_size=256,
                stride=128,
                epochs=3,
                num_inputs=2,
                best_batch_size=4,
            )
            tuner.run_trial(
                trial_two,
                train_samples=["train-a"],
                val_samples=["val-a"],
                patch_size=256,
                stride=128,
                epochs=3,
                num_inputs=2,
                best_batch_size=4,
            )

        cache_files = [
            call.kwargs["cache_file"] for call in build_dataset.call_args_list
        ]
        first_trial_caches = cache_files[:2]
        second_trial_caches = cache_files[2:]

        self.assertEqual(first_trial_caches, second_trial_caches)
        self.assertNotEqual(first_trial_caches[0], first_trial_caches[1])
        self.assertNotIn("trial", first_trial_caches[0])
        self.assertTrue(build_dataset.call_args_list[0].kwargs["augment"])
        self.assertFalse(build_dataset.call_args_list[1].kwargs["augment"])

    def test_cv_tuner_run_trial_marks_trial_failed_on_oom(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        tuner = train.CVTuner()
        oom_model = SimpleNamespace(
            compile=Mock(),
            fit=Mock(side_effect=train.tf.errors.ResourceExhaustedError("oom")),
        )
        tuner.hypermodel = SimpleNamespace(build=lambda _hp: oom_model)

        with (
            patch.object(
                train, "build_dataset", side_effect=[[object(), object()], [object()]]
            ),
            patch.object(
                train.tf.keras.callbacks,
                "EarlyStopping",
                side_effect=lambda **kwargs: ("early-stopping", kwargs),
            ),
        ):
            result = tuner.run_trial(
                SimpleNamespace(
                    trial_id="001",
                    hyperparameters=_FakeTrialHyperparameters(
                        {"dropout": 0.2, "learning_rate": 1e-3}
                    ),
                ),
                train_samples=["train-a"],
                val_samples=["val-a"],
                patch_size=256,
                stride=128,
                epochs=3,
                num_inputs=2,
                best_batch_size=4,
            )

        self.assertEqual(result, {"val_loss": 1e9})

    def test_train_model_uses_validation_holdout_for_tuned_final_training(self) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_train_dataset = [object(), object(), object()]
        final_val_dataset = [object()]
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)
        buffer = io.StringIO()

        with (
            patch.object(
                train.tf.keras.callbacks,
                "EarlyStopping",
                side_effect=lambda **kwargs: ("early-stopping", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "TensorBoard",
                side_effect=lambda **kwargs: ("tensorboard", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ReduceLROnPlateau",
                side_effect=lambda **kwargs: ("reduce-lr", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ModelCheckpoint",
                side_effect=lambda **kwargs: ("checkpoint", kwargs),
            ),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_holdout_split",
                return_value=(["train-a"], ["val-a"]),
            ) as create_spatial_holdout_split,
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner) as create_tuner,
            patch.object(
                train,
                "build_dataset",
                side_effect=[final_train_dataset, final_val_dataset],
            ),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
        ):
            with redirect_stdout(buffer):
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
                    run_name="holdout-training",
                    tuning_dir="/tmp/tuning",
                    validation_fraction=0.2,
                    random_state=42,
                    use_mixed_precision=False,
                    max_trials=3,
                    skip_tuning=False,
                )

        create_spatial_holdout_split.assert_called_once_with(
            ["sample-a", "sample-b"],
            tile_size=256,
            validation_fraction=0.2,
            random_state=42,
            coverage_bins=8,
        )
        self.assertEqual(create_tuner.call_args.kwargs["validation_fraction"], 0.2)
        tuner.search.assert_called_once()
        final_model.fit.assert_called_once()
        self.assertEqual(final_model.fit.call_args.kwargs["epochs"], 100)
        self.assertEqual(
            final_model.fit.call_args.kwargs["validation_data"],
            final_val_dataset,
        )
        self.assertEqual(
            final_model.fit.call_args.kwargs["callbacks"][0][0], "early-stopping"
        )
        self.assertEqual(
            final_model.fit.call_args.kwargs["callbacks"][0][1]["monitor"], "val_loss"
        )
        self.assertIn("monitor_metric         : val_loss", buffer.getvalue())
        self.assertNotIn("Frozen CV epoch selection", buffer.getvalue())

    def test_train_model_resuming_without_checkpoint_uses_validation_holdout(
        self,
    ) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_train_dataset = [object(), object(), object()]
        final_val_dataset = [object()]
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)

        with (
            patch.object(
                train.tf.keras.callbacks,
                "TensorBoard",
                side_effect=lambda **kwargs: ("tensorboard", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ReduceLROnPlateau",
                side_effect=lambda **kwargs: ("reduce-lr", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ModelCheckpoint",
                side_effect=lambda **kwargs: ("checkpoint", kwargs),
            ),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_holdout_split",
                return_value=(["train-a"], ["val-a"]),
            ),
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner),
            patch.object(
                train,
                "build_dataset",
                side_effect=[final_train_dataset, final_val_dataset],
            ),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
            patch.object(train, "infer_initial_epoch", return_value=3),
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
                run_name="holdout-training",
                tuning_dir="/tmp/tuning",
                validation_fraction=0.2,
                random_state=42,
                use_mixed_precision=False,
                max_trials=3,
                skip_tuning=False,
            )

        final_model.fit.assert_called_once()
        self.assertEqual(final_model.fit.call_args.kwargs["epochs"], 100)
        self.assertEqual(
            final_model.fit.call_args.kwargs["validation_data"],
            final_val_dataset,
        )

    def test_train_model_skips_fit_when_resume_already_meets_epoch_budget(
        self,
    ) -> None:
        _install_training_import_stubs()
        train = _reload_module("train")
        best_hp = _FakeHyperparameters({"dropout": 0.0, "learning_rate": 1e-3})
        tuner = SimpleNamespace(
            search=Mock(),
            get_best_hyperparameters=Mock(return_value=[best_hp]),
        )
        final_train_dataset = [object(), object(), object()]
        final_val_dataset = [object()]
        final_model = SimpleNamespace(fit=Mock(), save=Mock(), optimizer=None)

        with (
            patch.object(
                train.tf.keras.callbacks,
                "TensorBoard",
                side_effect=lambda **kwargs: ("tensorboard", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ReduceLROnPlateau",
                side_effect=lambda **kwargs: ("reduce-lr", kwargs),
            ),
            patch.object(
                train.tf.keras.callbacks,
                "ModelCheckpoint",
                side_effect=lambda **kwargs: ("checkpoint", kwargs),
            ),
            patch.object(train.tf.config, "list_logical_devices", return_value=[]),
            patch.object(train, "print_training_image_paths"),
            patch.object(train, "list_samples", return_value=["sample-a", "sample-b"]),
            patch.object(
                train,
                "create_spatial_holdout_split",
                return_value=(["train-a"], ["val-a"]),
            ),
            patch.object(train, "find_optimal_batch_size", return_value=2),
            patch.object(train, "create_tuner", return_value=tuner),
            patch.object(
                train,
                "build_dataset",
                side_effect=[final_train_dataset, final_val_dataset],
            ),
            patch.object(train, "build_final_model", return_value=final_model),
            patch.object(train, "compile_model_for_training", return_value=final_model),
            patch.object(train, "infer_initial_epoch", return_value=100),
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
                run_name="holdout-training",
                tuning_dir="/tmp/tuning",
                validation_fraction=0.2,
                random_state=42,
                use_mixed_precision=False,
                max_trials=3,
                skip_tuning=False,
            )

        final_model.fit.assert_not_called()
        final_model.save.assert_called_once_with("model.keras")


if __name__ == "__main__":
    unittest.main()
