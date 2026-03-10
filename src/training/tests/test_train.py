import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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


def _install_training_import_stubs() -> None:
    np_module = types.ModuleType("numpy")
    tf_module = types.ModuleType("tensorflow")
    tf_module.errors = SimpleNamespace(ResourceExhaustedError=RuntimeError)
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


if __name__ == "__main__":
    unittest.main()
