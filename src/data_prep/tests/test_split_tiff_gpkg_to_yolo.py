import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import tifffile
from shapely.geometry import MultiPolygon, Polygon


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "split_tiff_gpkg_to_yolo.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        raise AssertionError(f"Missing script under test: {SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("split_tiff_gpkg_to_yolo", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stems(paths: list[Path]) -> set[str]:
    return {path.stem for path in paths}


class SplitTiffGpkgToYoloTests(unittest.TestCase):
    def test_cli_writes_train_val_images_and_labels(self) -> None:
        module = _load_module()
        image = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.tif"
            polygons_path = tmp_path / "polygons.gpkg"
            output_dir = tmp_path / "output"

            tifffile.imwrite(image_path, image)
            gdf = gpd.GeoDataFrame(
                {"geometry": [Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]}
            )
            gdf.to_file(polygons_path, layer="grains", driver="GPKG")

            argv = [
                "split_tiff_gpkg_to_yolo.py",
                "--image",
                str(image_path),
                "--polygons",
                str(polygons_path),
                "--output-dir",
                str(output_dir),
                "--patch-size",
                "4",
                "--stride",
                "4",
                "--tile-size",
                "4",
                "--validation-fraction",
                "0.5",
                "--random-state",
                "7",
            ]
            with mock.patch("sys.argv", argv):
                module.main()

            train_images = sorted((output_dir / "images" / "train").glob("*.tif"))
            val_images = sorted((output_dir / "images" / "val").glob("*.tif"))
            train_labels = sorted((output_dir / "labels" / "train").glob("*.txt"))
            val_labels = sorted((output_dir / "labels" / "val").glob("*.txt"))

            self.assertTrue(train_images)
            self.assertTrue(val_images)
            self.assertSetEqual(_stems(train_images), _stems(train_labels))
            self.assertSetEqual(_stems(val_images), _stems(val_labels))
            self.assertTrue((output_dir / "labels" / "train").exists())
            self.assertTrue((output_dir / "labels" / "val").exists())

            with tifffile.TiffFile(train_images[0]) as tif:
                self.assertEqual(tif.series[0].axes, "CYX")
                written_patch = tif.asarray()
                self.assertEqual(written_patch.shape, (4, 4, 4))
                self.assertEqual(written_patch.dtype, np.uint8)

            all_label_paths = train_labels + val_labels
            self.assertTrue(all_label_paths)
            self.assertTrue(
                any(label_path.read_text().strip() for label_path in all_label_paths)
            )

    def test_build_yolo_rows_normalizes_padded_edge_patch_against_saved_patch_size(
        self,
    ) -> None:
        module = _load_module()
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        rows = module.build_yolo_rows(
            [polygon],
            patch_bounds=(0, 2, 0, 2),
            patch_size=4,
        )

        self.assertEqual(rows, ["0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5"])

    def test_negative_y_polygons_are_flipped_into_positive_image_space(self) -> None:
        module = _load_module()
        polygons = [Polygon([(0, -2), (2, -2), (2, 0), (0, 0)])]

        normalized = module._normalize_polygons_to_image_space(polygons)
        rows = module.build_yolo_rows(
            normalized,
            patch_bounds=(0, 4, 0, 4),
            patch_size=4,
        )

        self.assertEqual(rows, ["0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5"])

    def test_normalize_image_ext_rejects_non_tiff_extensions(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(
            ValueError,
            r"image_ext must be one of \.tif or \.tiff",
        ):
            module._normalize_image_ext(".png")

    def test_export_constrains_region_0000_patch_when_tile_is_smaller_than_patch(
        self,
    ) -> None:
        module = _load_module()
        image = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.tif"
            polygons_path = tmp_path / "polygons.gpkg"
            output_dir = tmp_path / "output"

            tifffile.imwrite(image_path, image, metadata={"axes": "CYX"})
            gdf = gpd.GeoDataFrame(
                {
                    "geometry": [
                        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
                        Polygon([(0, 2), (2, 2), (2, 4), (0, 4)]),
                    ]
                }
            )
            gdf.to_file(polygons_path, layer="grains", driver="GPKG")

            module.export_dataset(
                image_path=image_path,
                polygons_path=polygons_path,
                output_dir=output_dir,
                patch_size=4,
                stride=2,
                tile_size=2,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
                image_ext=".tif",
            )

            region_0000_label = (
                output_dir / "labels" / "train" / "region_0000_y00000_x00000.txt"
            )
            self.assertTrue(region_0000_label.exists())
            self.assertEqual(region_0000_label.read_text(), "")

            region_0000_image = (
                output_dir / "images" / "train" / "region_0000_y00000_x00000.tif"
            )
            with tifffile.TiffFile(region_0000_image) as tif:
                self.assertEqual(tif.series[0].axes, "CYX")
                patch = tif.asarray()

            self.assertEqual(patch.shape, (2, 4, 4))
            np.testing.assert_array_equal(patch[:, :2, :2], image[:, :2, :2])
            np.testing.assert_array_equal(patch[:, 2:, :], np.zeros((2, 2, 4), dtype=np.uint8))
            np.testing.assert_array_equal(patch[:, :, 2:], np.zeros((2, 4, 2), dtype=np.uint8))

    def test_export_clears_stale_split_files_when_rerun(self) -> None:
        module = _load_module()
        image = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
        coverages = np.ones(4, dtype=np.float32)
        first_train, first_val = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=0,
            coverage_bins=4,
        )
        second_train, second_val = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=1,
            coverage_bins=4,
        )
        self.assertFalse(
            np.array_equal(first_train, second_train)
            and np.array_equal(first_val, second_val)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.tif"
            polygons_path = tmp_path / "polygons.gpkg"
            output_dir = tmp_path / "output"

            tifffile.imwrite(image_path, image, metadata={"axes": "CYX"})
            gdf = gpd.GeoDataFrame(
                {
                    "geometry": [
                        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
                        Polygon([(0, 2), (2, 2), (2, 4), (0, 4)]),
                        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
                    ]
                }
            )
            gdf.to_file(polygons_path, layer="grains", driver="GPKG")

            module.export_dataset(
                image_path=image_path,
                polygons_path=polygons_path,
                output_dir=output_dir,
                patch_size=2,
                stride=2,
                tile_size=2,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
                image_ext=".tif",
            )
            module.export_dataset(
                image_path=image_path,
                polygons_path=polygons_path,
                output_dir=output_dir,
                patch_size=2,
                stride=2,
                tile_size=2,
                validation_fraction=0.5,
                random_state=1,
                coverage_bins=4,
                image_ext=".tif",
            )

            train_images = sorted((output_dir / "images" / "train").glob("*.tif"))
            val_images = sorted((output_dir / "images" / "val").glob("*.tif"))
            train_labels = sorted((output_dir / "labels" / "train").glob("*.txt"))
            val_labels = sorted((output_dir / "labels" / "val").glob("*.txt"))

        train_stems = _stems(train_images)
        val_stems = _stems(val_images)
        self.assertSetEqual(train_stems, _stems(train_labels))
        self.assertSetEqual(val_stems, _stems(val_labels))
        self.assertTrue(train_stems.isdisjoint(val_stems))

    def test_polygon_intersection_writes_normalized_yolo_segmentation_row(self) -> None:
        module = _load_module()
        polygon = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
        patch_bounds = (4, 8, 4, 8)

        rows = module.build_yolo_rows(
            [polygon],
            patch_bounds=patch_bounds,
            patch_size=4,
        )

        self.assertEqual(rows, ["0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5"])

    def test_build_yolo_rows_skips_polygons_outside_patch(self) -> None:
        module = _load_module()
        polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

        rows = module.build_yolo_rows(
            [polygon],
            patch_bounds=(4, 8, 4, 8),
            patch_size=4,
        )

        self.assertEqual(rows, [])

    def test_build_yolo_rows_skips_invalid_input_polygon_without_raising(self) -> None:
        module = _load_module()
        polygon = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
        self.assertFalse(polygon.is_valid)

        rows = module.build_yolo_rows(
            [polygon],
            patch_bounds=(0, 4, 0, 4),
            patch_size=4,
        )

        self.assertEqual(rows, [])

    def test_build_yolo_rows_emits_one_row_per_multipart_polygon_part(self) -> None:
        module = _load_module()
        polygons = [
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
                ]
            )
        ]

        rows = module.build_yolo_rows(
            polygons,
            patch_bounds=(0, 4, 0, 4),
            patch_size=4,
        )

        self.assertCountEqual(
            rows,
            [
                "0 0.0 0.0 0.25 0.0 0.25 0.25 0.0 0.25",
                "0 0.75 0.75 1.0 0.75 1.0 1.0 0.75 1.0",
            ],
        )

    def test_compute_starts_hits_last_possible_patch_origin(self) -> None:
        module = _load_module()

        starts = module.compute_starts(size=10, patch_size=4, stride=3)

        self.assertEqual(starts, [0, 3, 6])

    def test_compute_starts_appends_edge_covering_start(self) -> None:
        module = _load_module()

        starts = module.compute_starts(size=11, patch_size=4, stride=3)

        self.assertEqual(starts, [0, 3, 6, 7])

    def test_compute_starts_returns_zero_for_small_regions(self) -> None:
        module = _load_module()

        starts = module.compute_starts(size=3, patch_size=4, stride=2)

        self.assertEqual(starts, [0])

    def test_compute_starts_returns_zero_when_size_equals_patch_size(self) -> None:
        module = _load_module()

        starts = module.compute_starts(size=4, patch_size=4, stride=2)

        self.assertEqual(starts, [0])

    def test_compute_starts_raises_when_stride_exceeds_patch_size(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "stride must not exceed patch_size"):
            module.compute_starts(size=20, patch_size=4, stride=10)

    def test_compute_starts_raises_when_patch_size_is_not_positive(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "patch_size must be > 0"):
            module.compute_starts(size=20, patch_size=0, stride=4)

    def test_compute_starts_raises_when_patch_size_is_negative(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "patch_size must be > 0"):
            module.compute_starts(size=20, patch_size=-1, stride=4)

    def test_compute_starts_raises_when_stride_is_not_positive(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "stride must be > 0"):
            module.compute_starts(size=20, patch_size=4, stride=0)

    def test_compute_starts_raises_when_stride_is_negative(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "stride must be > 0"):
            module.compute_starts(size=20, patch_size=4, stride=-1)

    def test_save_patch_writes_uint8_tiff_with_channel_first_axes(
        self,
    ) -> None:
        module = _load_module()
        image = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patch.tif"

            module.save_patch(output_path, image)

            with tifffile.TiffFile(output_path) as tif:
                self.assertEqual(tif.series[0].axes, "CYX")
                written = tif.asarray()

        np.testing.assert_array_equal(written, image.astype(np.uint8))
        self.assertEqual(written.dtype, np.uint8)

    def test_save_patch_clips_values_above_255_before_casting_to_uint8(self) -> None:
        module = _load_module()
        image = np.array([[[0, 255, 256, 511]]], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "patch.tif"

            module.save_patch(output_path, image)
            written = tifffile.imread(output_path)

        np.testing.assert_array_equal(
            written,
            np.array([[[0, 255, 255, 255]]], dtype=np.uint8),
        )

    def test_load_image_keeps_channel_first_input_unchanged(self) -> None:
        module = _load_module()
        image = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "channel_first.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "SYX"})

            loaded = module.load_image_channel_first(image_path)

        self.assertEqual(loaded.shape, (4, 5, 6))
        np.testing.assert_array_equal(loaded, image)

    def test_load_image_transposes_channel_last_input_once(self) -> None:
        module = _load_module()
        image = np.arange(5 * 6 * 4, dtype=np.uint8).reshape(5, 6, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "channel_last.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "YXS"})

            loaded = module.load_image_channel_first(image_path)

        self.assertEqual(loaded.shape, (4, 5, 6))
        np.testing.assert_array_equal(loaded, np.transpose(image, (2, 0, 1)))

    def test_load_image_keeps_explicit_cyx_input_unchanged(self) -> None:
        module = _load_module()
        image = np.arange(5 * 8 * 9, dtype=np.uint8).reshape(5, 8, 9)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "explicit_cyx.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "CYX"})

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "CYX")

            loaded = module.load_image_channel_first(image_path)

        self.assertEqual(loaded.shape, (5, 8, 9))
        np.testing.assert_array_equal(loaded, image)

    def test_load_image_keeps_qyx_channel_first_input_unchanged(self) -> None:
        module = _load_module()
        image = np.arange(5 * 8 * 9, dtype=np.uint8).reshape(5, 8, 9)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "qyx_channel_first.tif"
            tifffile.imwrite(image_path, image)

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "QYX")

            loaded = module.load_image_channel_first(image_path)

        self.assertEqual(loaded.shape, (5, 8, 9))
        np.testing.assert_array_equal(loaded, image)

    def test_load_image_transposes_qyx_channel_last_input(self) -> None:
        module = _load_module()
        image = np.arange(10 * 8 * 5, dtype=np.uint8).reshape(10, 8, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "qyx_channel_last.tif"
            tifffile.imwrite(image_path, image)

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "QYX")

            loaded = module.load_image_channel_first(image_path)

        self.assertEqual(loaded.shape, (5, 10, 8))
        np.testing.assert_array_equal(loaded, np.transpose(image, (2, 0, 1)))

    def test_load_image_raises_for_ambiguous_qyx_layout(self) -> None:
        module = _load_module()
        image = np.arange(8 * 9 * 8, dtype=np.uint8).reshape(8, 9, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "qyx_ambiguous.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "QYX"})

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "QYX")

            with self.assertRaisesRegex(
                ValueError,
                "cannot be inferred safely",
            ):
                module.load_image_channel_first(image_path)

    def test_load_image_raises_for_non_equal_ambiguous_qyx_layout(self) -> None:
        module = _load_module()
        image = np.arange(5 * 10 * 6, dtype=np.uint8).reshape(5, 10, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "qyx_ambiguous_non_equal.tif"
            tifffile.imwrite(image_path, image)

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "QYX")

            with self.assertRaisesRegex(
                ValueError,
                "cannot be inferred safely",
            ):
                module.load_image_channel_first(image_path)

    def test_load_image_raises_for_unrecognized_3d_axes(self) -> None:
        module = _load_module()
        image = np.arange(5 * 8 * 9, dtype=np.uint8).reshape(5, 8, 9)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "unknown_axes.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "ZYX"})

            with tifffile.TiffFile(image_path) as tif:
                self.assertEqual(tif.series[0].axes, "ZYX")

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported 3D TIFF axes",
            ):
                module.load_image_channel_first(image_path)

    def test_regions_below_min_validation_coverage_stay_in_train(self) -> None:
        module = _load_module()
        coverages = np.array([0.0, 0.05, 0.15, 0.30], dtype=np.float32)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=7,
            coverage_bins=4,
        )

        self.assertIn(0, train_idx)
        self.assertIn(1, train_idx)
        self.assertTrue(set(train_idx).isdisjoint(val_idx))

    def test_split_region_indices_is_deterministic_for_fixed_seed(self) -> None:
        module = _load_module()
        coverages = np.array([0.0, 0.12, 0.18, 0.25, 0.31], dtype=np.float32)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=7,
            coverage_bins=4,
        )
        second_train_idx, second_val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=7,
            coverage_bins=4,
        )

        np.testing.assert_array_equal(train_idx, second_train_idx)
        np.testing.assert_array_equal(val_idx, second_val_idx)
        np.testing.assert_array_equal(train_idx, np.array([0, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(val_idx, np.array([1, 3], dtype=np.int64))

    def test_region_with_exact_min_validation_coverage_is_eligible(self) -> None:
        module = _load_module()
        coverages = np.array([0.10, 0.20, 0.05], dtype=np.float32)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=0,
            coverage_bins=4,
        )

        self.assertIn(0, val_idx)
        self.assertIn(2, train_idx)
        self.assertTrue(set(train_idx).isdisjoint(val_idx))

    def test_split_region_indices_raises_with_fewer_than_two_eligible_regions(
        self,
    ) -> None:
        module = _load_module()
        coverages = np.array([0.0, 0.09, 0.10], dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError,
            "Not enough validation-eligible regions",
        ):
            module.split_region_indices(
                coverages,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
            )

    def test_split_region_indices_uses_ceil_for_validation_count(self) -> None:
        module = _load_module()
        coverages = np.array([0.11, 0.12, 0.13, 0.0], dtype=np.float32)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.34,
            random_state=0,
            coverage_bins=4,
        )

        np.testing.assert_array_equal(train_idx, np.array([1, 3], dtype=np.int64))
        np.testing.assert_array_equal(val_idx, np.array([0, 2], dtype=np.int64))

    def test_split_region_indices_clamps_validation_count_below_total_eligible(
        self,
    ) -> None:
        module = _load_module()
        coverages = np.array([0.11, 0.12, 0.13, 0.0], dtype=np.float32)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.99,
            random_state=0,
            coverage_bins=4,
        )

        np.testing.assert_array_equal(train_idx, np.array([1, 3], dtype=np.int64))
        np.testing.assert_array_equal(val_idx, np.array([0, 2], dtype=np.int64))

    def test_split_region_indices_raises_for_non_finite_coverages(self) -> None:
        module = _load_module()
        coverages = np.array([0.12, np.nan, 0.24], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "coverages must be finite"):
            module.split_region_indices(
                coverages,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
            )

    def test_split_region_indices_raises_for_non_1d_coverages(self) -> None:
        module = _load_module()
        coverages = np.array([[0.12, 0.24], [0.36, 0.48]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "coverages must be 1D"):
            module.split_region_indices(
                coverages,
                validation_fraction=0.5,
                random_state=0,
                coverage_bins=4,
            )

    def test_split_region_indices_preserves_near_threshold_precision(self) -> None:
        module = _load_module()
        just_below = np.nextafter(0.10, 0.0)
        just_above = np.nextafter(0.10, 1.0)
        coverages = np.array([just_below, just_above, 0.20], dtype=np.float64)

        train_idx, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=0.5,
            random_state=0,
            coverage_bins=4,
        )

        np.testing.assert_array_equal(train_idx, np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(val_idx, np.array([1], dtype=np.int64))

    def test_split_region_indices_uses_coverage_bins_when_stratification_is_feasible(
        self,
    ) -> None:
        module = _load_module()
        coverages = np.array([0.11, 0.12, 0.13, 0.81, 0.82, 0.83], dtype=np.float32)

        _, val_idx = module.split_region_indices(
            coverages,
            validation_fraction=1 / 3,
            random_state=2,
            coverage_bins=2,
        )

        val_coverages = coverages[val_idx]
        self.assertTrue(np.any(val_coverages < 0.5))
        self.assertTrue(np.any(val_coverages > 0.5))
