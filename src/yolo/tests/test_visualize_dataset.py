import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _write_rgb_tiff(path: Path) -> None:
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[..., 0] = 120
    array[..., 1] = 80
    array[..., 2] = 40
    Image.fromarray(array, mode="RGB").save(path)


def _write_multiframe_tiff(path: Path) -> None:
    frames = []
    for value in (32, 128, 224):
        frame = np.full((32, 32), value, dtype=np.uint8)
        frames.append(Image.fromarray(frame, mode="L"))
    frames[0].save(path, save_all=True, append_images=frames[1:])


def _write_label(path: Path) -> None:
    path.write_text(
        "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n",
        encoding="utf-8",
    )


def _build_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    _write_rgb_tiff(root / "images" / "train" / "sample_train.tif")
    _write_label(root / "labels" / "train" / "sample_train.txt")

    _write_multiframe_tiff(root / "images" / "val" / "sample_val.tif")
    _write_label(root / "labels" / "val" / "sample_val.txt")

    dataset_yaml = root / "toy.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: grain\n",
        encoding="utf-8",
    )
    return dataset_yaml


class VisualizeDatasetTests(unittest.TestCase):
    def test_parse_args_accepts_dataset_dir_and_num(self) -> None:
        module = _reload_module("visualize_dataset")

        args = module.parse_args(["/tmp/dataset", "-n", "3"])

        self.assertEqual(args.dataset_dir, Path("/tmp/dataset"))
        self.assertEqual(args.num, 3)
        self.assertEqual(args.output_dir, Path("/tmp/dataset") / "visualizations")

    def test_collect_samples_keeps_empty_label_files(self) -> None:
        module = _reload_module("visualize_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)
            (dataset_dir / "images" / "val").mkdir(parents=True)
            (dataset_dir / "labels" / "val").mkdir(parents=True)

            _write_rgb_tiff(dataset_dir / "images" / "train" / "empty_label.tif")
            (dataset_dir / "labels" / "train" / "empty_label.txt").write_text(
                "",
                encoding="utf-8",
            )
            (dataset_dir / "toy.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: grain\n",
                encoding="utf-8",
            )

            dataset_root, config, _ = module.load_dataset_config(dataset_dir)
            samples = module.collect_samples(dataset_root, config, "train")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][0].name, "empty_label.tif")
        self.assertEqual(samples[0][1].name, "empty_label.txt")

    def test_read_polygons_scales_normalized_points(self) -> None:
        module = _reload_module("visualize_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "sample.txt"
            _write_label(label_path)

            polygons = module._read_polygons(
                label_path,
                image_width=100,
                image_height=80,
            )

        self.assertEqual(len(polygons), 1)
        class_id, points = polygons[0]
        self.assertEqual(class_id, 0)
        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.allclose(points[0], np.array([20.0, 16.0])))

    def test_main_saves_train_and_val_visualizations(self) -> None:
        module = _reload_module("visualize_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "output"
            _build_dataset(dataset_dir)

            module.main(
                [
                    str(dataset_dir),
                    "--output-dir",
                    str(output_dir),
                    "--num",
                    "1",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(len(list((output_dir / "train").glob("*.png"))), 1)
            self.assertEqual(len(list((output_dir / "val").glob("*.png"))), 1)

    def test_main_prints_saved_counts(self) -> None:
        module = _reload_module("visualize_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "output"
            _build_dataset(dataset_dir)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                module.main(
                    [
                        str(dataset_dir),
                        "--output-dir",
                        str(output_dir),
                        "--num",
                        "1",
                    ]
                )

        output = buffer.getvalue()
        self.assertIn("Saved 1 visualization(s) for train", output)
        self.assertIn("Saved 1 visualization(s) for val", output)


if __name__ == "__main__":
    unittest.main()
