import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "raster_to_polygon.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"Missing script under test: {SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("raster_to_polygon", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


class RasterToPolygonTests(unittest.TestCase):
    def test_binary_mask_selects_only_requested_class(self) -> None:
        module = _load_module()

        raster = np.array(
            [
                [0, 1, 2, 1],
                [2, 1, 0, 2],
            ],
            dtype=np.uint8,
        )

        expected = np.array(
            [
                [0, 1, 0, 1],
                [0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )

        binary = module._binary_mask(raster, class_value=1)
        np.testing.assert_array_equal(binary, expected)

    def test_component_masks_split_disconnected_regions(self) -> None:
        module = _load_module()

        binary = np.array(
            [
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        component_masks = module._component_masks(binary)

        self.assertEqual(len(component_masks), 3)
        self.assertTrue(all(mask.dtype == np.uint8 for mask in component_masks))
        self.assertTrue(all(int(mask.max()) == 1 for mask in component_masks))

    def test_component_to_polygon_preserves_pixel_area_and_flip_convention(
        self,
    ) -> None:
        module = _load_module()

        component = np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )

        polygon = module._component_to_polygon(component, flip_y=True)
        polygon_no_flip = module._component_to_polygon(component, flip_y=False)

        self.assertEqual(polygon.area, 4.0)
        self.assertEqual(polygon.bounds, (1.0, -2.0, 3.0, 0.0))
        self.assertEqual(polygon_no_flip.area, 4.0)
        self.assertEqual(polygon_no_flip.bounds, (1.0, 0.0, 3.0, 2.0))

    def test_cli_writes_gpkg_with_one_polygon_per_interior_component(self) -> None:
        module = _load_module()

        raster = np.array(
            [
                [2, 2, 2, 0, 0, 0, 0],
                [2, 1, 1, 0, 1, 1, 0],
                [2, 1, 1, 0, 1, 1, 0],
                [2, 2, 2, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "prediction.png"
            output_path = tmp_path / "prediction_polygons.gpkg"
            _write_mask(input_path, raster)

            argv = [
                "raster_to_polygon.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--output-layer",
                "grains",
                "--class-value",
                "1",
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            self.assertTrue(output_path.exists())

            gdf = gpd.read_file(output_path, layer="grains")

            self.assertEqual(len(gdf), 2)
            self.assertTrue((gdf.geometry.type == "Polygon").all())
            self.assertTrue((gdf["class_value"] == 1).all())
            self.assertEqual(sorted(gdf["pixel_area"].tolist()), [4, 4])
            self.assertEqual(sorted(gdf.geometry.area.tolist()), [4.0, 4.0])
            self.assertTrue(gdf.total_bounds[3] <= 0.0)

    def test_cli_no_flip_y_preserves_positive_image_coordinates(self) -> None:
        module = _load_module()

        raster = np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "prediction.png"
            output_path = tmp_path / "prediction_polygons.gpkg"
            _write_mask(input_path, raster)

            argv = [
                "raster_to_polygon.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--output-layer",
                "grains",
                "--no-flip-y",
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            gdf = gpd.read_file(output_path, layer="grains")

            self.assertEqual(len(gdf), 1)
            self.assertEqual(gdf.geometry.iloc[0].bounds, (1.0, 0.0, 3.0, 2.0))

    def test_cli_writes_empty_layer_when_no_components_match(self) -> None:
        module = _load_module()

        raster = np.zeros((3, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "prediction.png"
            output_path = tmp_path / "prediction_polygons.gpkg"
            _write_mask(input_path, raster)

            argv = [
                "raster_to_polygon.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--output-layer",
                "grains",
                "--min-area",
                "10",
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            gdf = gpd.read_file(output_path, layer="grains")
            self.assertEqual(len(gdf), 0)


if __name__ == "__main__":
    unittest.main()
