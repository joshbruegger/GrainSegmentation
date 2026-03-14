import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import tifffile
from shapely.affinity import scale as scale_geometry
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

MIN_VALIDATION_COVERAGE = 0.10


def compute_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size")
    if size <= patch_size:
        return [0]

    last_start = size - patch_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def load_image_channel_first(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes

    if image.ndim != 3:
        return image

    if axes == "YXS":
        return np.transpose(image, (2, 0, 1))

    if axes in {"SYX", "CYX"}:
        return image

    if axes == "QYX":
        first, middle, last = image.shape

        if first < middle and last >= middle:
            return image

        if last < middle and first >= middle:
            return np.transpose(image, (2, 0, 1))

        raise ValueError(
            f"TIFF layout cannot be inferred safely from axes={axes!r} and shape={image.shape}"
        )

    raise ValueError(f"Unsupported 3D TIFF axes {axes!r} for shape {image.shape}")


def save_patch(path: Path, patch: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        np.clip(patch, 0, 255).astype(np.uint8, copy=False),
        metadata={"axes": "CYX"},
    )


def _format_yolo_value(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def _iter_polygon_parts(
    geometry: Polygon | MultiPolygon | GeometryCollection,
) -> list[Polygon]:
    if geometry.is_empty:
        return []

    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return []

    if isinstance(cleaned, Polygon):
        return [cleaned]

    if isinstance(cleaned, MultiPolygon):
        return [part for part in cleaned.geoms if not part.is_empty]

    if isinstance(cleaned, GeometryCollection):
        return [
            part
            for part in cleaned.geoms
            if isinstance(part, Polygon) and not part.is_empty
        ]

    return []


def _normalized_exterior_coordinates(polygon: Polygon) -> list[tuple[float, float]]:
    coordinates = list(orient(polygon, sign=1.0).exterior.coords[:-1])
    start_idx = min(range(len(coordinates)), key=lambda idx: coordinates[idx])
    return coordinates[start_idx:] + coordinates[:start_idx]


def build_yolo_rows(
    polygons: list[Polygon | MultiPolygon],
    patch_bounds: tuple[int, int, int, int],
    patch_size: int,
) -> list[str]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    y0, y1, x0, x1 = patch_bounds
    patch_height = y1 - y0
    patch_width = x1 - x0
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch_bounds must define a positive-area patch")
    if patch_height > patch_size or patch_width > patch_size:
        raise ValueError("patch_bounds dimensions must not exceed patch_size")

    patch_box = box(x0, y0, x1, y1)
    rows: list[str] = []
    for polygon in polygons:
        if polygon.is_empty or not polygon.is_valid:
            continue

        try:
            clipped = polygon.intersection(patch_box)
        except GEOSException:
            continue

        for part in _iter_polygon_parts(clipped):
            coordinates = _normalized_exterior_coordinates(part)
            distinct_points = list(dict.fromkeys(coordinates))
            if len(distinct_points) < 3:
                continue

            normalized = []
            for x, y in coordinates:
                normalized.append(_format_yolo_value((x - x0) / patch_size))
                normalized.append(_format_yolo_value((y - y0) / patch_size))
            rows.append("0 " + " ".join(normalized))
    return rows


def split_region_indices(
    coverages: np.ndarray,
    validation_fraction: float,
    random_state: int,
    coverage_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    coverages_arr = np.asarray(coverages)
    if coverages_arr.ndim != 1:
        raise ValueError("coverages must be 1D before splitting.")

    if not np.all(np.isfinite(coverages_arr)):
        raise ValueError("coverages must be finite before splitting.")

    eligible_indices = np.flatnonzero(coverages_arr >= MIN_VALIDATION_COVERAGE)
    train_only_indices = np.flatnonzero(coverages_arr < MIN_VALIDATION_COVERAGE)
    total_eligible = eligible_indices.size

    if total_eligible < 2:
        raise ValueError(
            "Not enough validation-eligible regions "
            f"({total_eligible}) to create a train/validation split."
        )

    val_count = math.ceil(total_eligible * validation_fraction)
    val_count = max(1, min(val_count, total_eligible - 1))

    shuffled_eligible = eligible_indices.copy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(shuffled_eligible)

    eligible_coverages = coverages_arr[eligible_indices]
    group_count = max(2, math.ceil(1 / validation_fraction))
    bin_ids = _build_coverage_bin_ids(
        eligible_coverages,
        coverage_bins=coverage_bins,
        group_count=group_count,
    )
    counts = np.bincount(bin_ids)
    can_stratify = (
        counts.min() >= 2
        and np.count_nonzero(counts) <= val_count
        and np.count_nonzero(counts) <= (total_eligible - val_count)
    )

    if can_stratify:
        bin_by_index = {
            int(region_idx): int(bin_id)
            for region_idx, bin_id in zip(eligible_indices, bin_ids, strict=True)
        }
        ordered_bins: list[int] = []
        bin_queues: dict[int, list[int]] = {}
        for region_idx in shuffled_eligible:
            bin_id = bin_by_index[int(region_idx)]
            if bin_id not in bin_queues:
                bin_queues[bin_id] = []
                ordered_bins.append(bin_id)
            bin_queues[bin_id].append(int(region_idx))

        selected_val: list[int] = []
        for bin_id in ordered_bins:
            selected_val.append(bin_queues[bin_id].pop(0))

        while len(selected_val) < val_count:
            progressed = False
            for bin_id in ordered_bins:
                queue = bin_queues[bin_id]
                if len(queue) > 1 and len(selected_val) < val_count:
                    selected_val.append(queue.pop(0))
                    progressed = True
            if not progressed:
                break

        if len(selected_val) < val_count:
            for region_idx in shuffled_eligible:
                region_idx_int = int(region_idx)
                if region_idx_int not in selected_val:
                    selected_val.append(region_idx_int)
                if len(selected_val) == val_count:
                    break

        val_indices = np.sort(np.asarray(selected_val, dtype=np.int64))
    else:
        val_indices = np.sort(shuffled_eligible[:val_count])

    train_indices = np.sort(
        np.concatenate(
            [
                train_only_indices,
                np.setdiff1d(eligible_indices, val_indices, assume_unique=True),
            ]
        )
    )
    return train_indices, val_indices


def _build_coverage_bin_ids(
    coverages_arr: np.ndarray,
    *,
    coverage_bins: int,
    group_count: int,
) -> np.ndarray:
    total = coverages_arr.shape[0]
    if total <= 0:
        raise ValueError("At least one region is required for splitting.")

    if group_count <= 0:
        raise ValueError("group_count must be > 0")

    bins = min(coverage_bins, max(1, total // group_count))
    if bins <= 1:
        return np.zeros(total, dtype=int)

    bin_edges = np.quantile(coverages_arr, np.linspace(0.0, 1.0, bins + 1))
    if np.allclose(bin_edges, bin_edges[0]):
        return np.zeros(total, dtype=int)

    return np.digitize(coverages_arr, bin_edges[1:-1], right=True)


def _parse_patch_overlap(value: str) -> float:
    overlap = float(value)
    if overlap < 0.0 or overlap > 0.9:
        raise argparse.ArgumentTypeError("patch_overlap must be between 0.0 and 0.9")
    return overlap


def _stride_from_patch_overlap(patch_size: int, patch_overlap: float) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if patch_overlap < 0.0 or patch_overlap > 0.9:
        raise ValueError("patch_overlap must be between 0.0 and 0.9")
    return round(patch_size * (1 - patch_overlap))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patch-overlap", type=_parse_patch_overlap, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--coverage-bins", type=int, default=10)
    parser.add_argument("--image-ext", default=".tif")
    return parser.parse_args(argv)


def _normalize_image_ext(image_ext: str) -> str:
    normalized = image_ext if image_ext.startswith(".") else f".{image_ext}"
    if normalized not in {".tif", ".tiff"}:
        raise ValueError("image_ext must be one of .tif or .tiff")
    return normalized


def _load_polygons(path: Path) -> list[Polygon | MultiPolygon]:
    geodata = gpd.read_file(path)
    polygons: list[Polygon | MultiPolygon] = []
    for geometry in geodata.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if isinstance(geometry, (Polygon, MultiPolygon)):
            polygons.append(geometry)
            continue
        if isinstance(geometry, GeometryCollection):
            polygons.extend(
                part
                for part in geometry.geoms
                if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
            )
    return polygons


def _normalize_polygons_to_image_space(
    polygons: list[Polygon | MultiPolygon],
) -> list[Polygon | MultiPolygon]:
    if not polygons:
        return polygons

    min_y = min(polygon.bounds[1] for polygon in polygons if not polygon.is_empty)
    max_y = max(polygon.bounds[3] for polygon in polygons if not polygon.is_empty)
    if max_y <= 0 and min_y < 0:
        return [
            scale_geometry(polygon, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
            for polygon in polygons
        ]

    return polygons


def _region_bounds(
    height: int, width: int, tile_size: int
) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    bounds: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            bounds.append((y0, y1, x0, x1))
    return bounds


def _compute_region_coverages(
    region_bounds: list[tuple[int, int, int, int]],
    polygons: list[Polygon | MultiPolygon],
) -> np.ndarray:
    coverages = np.zeros(len(region_bounds), dtype=np.float32)
    for idx, (y0, y1, x0, x1) in enumerate(region_bounds):
        region_box = box(x0, y0, x1, y1)
        region_area = region_box.area
        if region_area == 0:
            continue

        covered_area = 0.0
        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue
            try:
                covered_area += polygon.intersection(region_box).area
            except GEOSException:
                continue
        coverages[idx] = min(covered_area / region_area, 1.0)
    return coverages


def _clear_output_files(directory: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def _prepare_output_dirs(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    split_dirs: dict[str, tuple[Path, Path]] = {}
    for split_name in ("train", "val"):
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        _clear_output_files(image_dir, ("*.tif", "*.tiff"))
        _clear_output_files(label_dir, ("*.txt",))
        split_dirs[split_name] = (image_dir, label_dir)
    return split_dirs


def _write_label_file(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def _extract_padded_patch(
    image: np.ndarray,
    patch_bounds: tuple[int, int, int, int],
    patch_size: int,
) -> np.ndarray:
    y0, y1, x0, x1 = patch_bounds
    patch = image[:, y0:y1, x0:x1]
    padded = np.zeros((image.shape[0], patch_size, patch_size), dtype=image.dtype)
    padded[:, : patch.shape[1], : patch.shape[2]] = patch
    return padded


def export_dataset(
    image_path: Path,
    polygons_path: Path,
    output_dir: Path,
    patch_size: int,
    stride: int,
    tile_size: int,
    validation_fraction: float,
    random_state: int,
    coverage_bins: int,
    image_ext: str,
) -> None:
    image = load_image_channel_first(image_path)
    polygons = _normalize_polygons_to_image_space(_load_polygons(polygons_path))
    _, height, width = image.shape

    region_bounds = _region_bounds(height, width, tile_size)
    coverages = _compute_region_coverages(region_bounds, polygons)
    train_indices, val_indices = split_region_indices(
        coverages,
        validation_fraction=validation_fraction,
        random_state=random_state,
        coverage_bins=coverage_bins,
    )

    split_dirs = _prepare_output_dirs(output_dir)
    normalized_image_ext = _normalize_image_ext(image_ext)

    for split_name, region_indices in (("train", train_indices), ("val", val_indices)):
        image_dir, label_dir = split_dirs[split_name]
        for region_idx in region_indices:
            y0, y1, x0, x1 = region_bounds[int(region_idx)]
            region_height = y1 - y0
            region_width = x1 - x0
            for y_offset in compute_starts(region_height, patch_size, stride):
                patch_y0 = y0 + y_offset
                patch_y1 = min(patch_y0 + patch_size, y1)
                for x_offset in compute_starts(region_width, patch_size, stride):
                    patch_x0 = x0 + x_offset
                    patch_x1 = min(patch_x0 + patch_size, x1)
                    patch_bounds = (patch_y0, patch_y1, patch_x0, patch_x1)
                    patch = _extract_padded_patch(image, patch_bounds, patch_size)
                    stem = (
                        f"region_{int(region_idx):04d}_y{patch_y0:05d}_x{patch_x0:05d}"
                    )
                    save_patch(image_dir / f"{stem}{normalized_image_ext}", patch)
                    rows = build_yolo_rows(
                        polygons,
                        patch_bounds=patch_bounds,
                        patch_size=patch_size,
                    )
                    _write_label_file(label_dir / f"{stem}.txt", rows)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    export_dataset(
        image_path=args.image,
        polygons_path=args.polygons,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=_stride_from_patch_overlap(
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
        ),
        tile_size=args.tile_size,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        coverage_bins=args.coverage_bins,
        image_ext=args.image_ext,
    )


if __name__ == "__main__":
    main()
