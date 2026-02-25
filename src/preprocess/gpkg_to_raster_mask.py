import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
except Exception as exc:  # pragma: no cover - optional dependency error
    rasterio = None
    rasterize = None
    from_bounds = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None


def _require_rasterio() -> None:
    if rasterio is None or rasterize is None or from_bounds is None:
        raise SystemExit(
            "rasterio is required for rasterization. "
            f"Import error: {_RASTERIO_IMPORT_ERROR}. "
            "Install it with: uv pip install rasterio"
        )


def _ensure_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom_types = gdf.geometry.geom_type
    if (geom_types == "MultiPolygon").any():
        gdf = gdf.explode(index_parts=False)
        gdf = gdf.reset_index(drop=True)
        geom_types = gdf.geometry.geom_type
    if not (geom_types == "Polygon").all():
        bad_types = sorted(set(geom_types[geom_types != "Polygon"]))
        raise SystemExit(f"Expected Polygon geometries, found: {', '.join(bad_types)}")
    return gdf


def _fix_invalid(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    return geom.buffer(0)


def _candidate_pairs(geoms: List) -> List[Tuple[int, int]]:
    series = gpd.GeoSeries(geoms)
    sindex = series.sindex
    if hasattr(sindex, "query_bulk"):
        try:
            left, right = sindex.query_bulk(series, predicate="intersects")
            mask = left < right
            return list(zip(left[mask], right[mask]))
        except Exception:
            pass
    pairs: List[Tuple[int, int]] = []
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        for j in sindex.intersection(geom.bounds):
            if j <= i:
                continue
            other = geoms[j]
            if other is None or other.is_empty:
                continue
            pairs.append((i, j))
    return pairs


def _resolve_overlaps_random(geoms: List, fix_invalid: bool) -> List:
    pairs = _candidate_pairs(geoms)
    rng = np.random.default_rng()
    for i, j in pairs:
        geom = geoms[i]
        other = geoms[j]
        if geom is None or geom.is_empty or other is None or other.is_empty:
            continue
        if not geom.intersects(other):
            continue
        overlap = geom.intersection(other)
        if overlap.is_empty:
            continue
        if rng.random() < 0.5:
            other = other.difference(overlap)
            if fix_invalid:
                other = _fix_invalid(other)
            geoms[j] = other
        else:
            geom = geom.difference(overlap)
            if fix_invalid:
                geom = _fix_invalid(geom)
            geoms[i] = geom
    return geoms


def _pixel_transform(width: int, height: int):
    bounds = (-0.5, height - 0.5, width - 0.5, -0.5)
    return from_bounds(*bounds, width, height)


def _load_reference_grid(
    reference: Optional[Path],
    bounds: Optional[Tuple[float, float, float, float]],
    width: Optional[int],
    height: Optional[int],
    pixel_coords: bool,
):
    if reference is not None:
        _require_rasterio()
        with rasterio.open(reference) as src:
            width = src.width
            height = src.height
            crs = src.crs
            transform = src.transform
        if pixel_coords:
            transform = _pixel_transform(width, height)
            crs = None
        return width, height, transform, crs

    if width is None or height is None:
        raise SystemExit("Provide --reference or --width/--height to define the grid.")

    if bounds is not None:
        _require_rasterio()
        transform = from_bounds(*bounds, width, height)
        crs = None
        return width, height, transform, crs

    _require_rasterio()
    transform = _pixel_transform(width, height)
    crs = None
    return width, height, transform, crs


def _split_boundary(geom, boundary_px: float, fix_invalid: bool):
    if geom is None or geom.is_empty:
        return None, None
    if fix_invalid:
        geom = _fix_invalid(geom)
    inner = geom.buffer(-boundary_px)
    if inner is None or inner.is_empty:
        return None, geom
    if fix_invalid:
        inner = _fix_invalid(inner)
    boundary = geom.difference(inner)
    if fix_invalid:
        boundary = _fix_invalid(boundary)
    return inner, boundary


def _build_shapes(
    gdf: gpd.GeoDataFrame,
    value_field: Optional[str],
    burn_value: int,
    fix_invalid: bool,
    boundary_px: float,
) -> Tuple[List[Tuple], List, int]:
    interior_shapes: List[Tuple] = []
    boundary_geoms: List = []
    max_value = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if value_field is not None:
            if value_field not in gdf.columns:
                raise SystemExit(f"Attribute '{value_field}' not found in input.")
            raw = row[value_field]
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"Attribute '{value_field}' must be numeric; got {raw!r}."
                )
        else:
            value = int(burn_value)

        interior, boundary = _split_boundary(geom, boundary_px, fix_invalid)
        if interior is not None and not interior.is_empty:
            interior_shapes.append((interior, value))
            max_value = max(max_value, value)
        if boundary is not None and not boundary.is_empty:
            boundary_geoms.append(boundary)
            max_value = max(max_value, value)
    return interior_shapes, boundary_geoms, max_value


def _choose_dtype(requested: Optional[str], max_value: int) -> str:
    if requested is not None:
        return requested
    if max_value <= np.iinfo(np.uint8).max:
        return "uint8"
    if max_value <= np.iinfo(np.uint16).max:
        return "uint16"
    return "uint32"


def _write_mask(mask: np.ndarray, output: Path, transform, crs) -> None:
    ext = output.suffix.lower()
    if ext in {".tif", ".tiff"}:
        _require_rasterio()
        with rasterio.open(
            output,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(mask, 1)
        return

    if mask.dtype not in (np.uint8, np.uint16):
        raise SystemExit(
            "Non-TIFF outputs support only uint8/uint16. "
            "Use --dtype uint8/uint16 or write a .tif."
        )
    image = Image.fromarray(mask)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize GeoPackage polygons into a U-Net training mask."
    )
    parser.add_argument("--input", required=True, help="Input GeoPackage path")
    parser.add_argument("--output", required=True, help="Output raster mask path")
    parser.add_argument("--layer", default=None, help="Layer name (default: first)")
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference raster to match (size/transform/CRS).",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Output bounds when no reference raster is provided.",
    )
    parser.add_argument("--width", type=int, help="Output width in pixels")
    parser.add_argument("--height", type=int, help="Output height in pixels")
    parser.add_argument(
        "--pixel-coords",
        action="store_true",
        help="Interpret polygons in pixel coordinates (x right, y down).",
    )
    parser.add_argument(
        "--attribute",
        default=None,
        help="Column name to burn into the mask (default: constant value).",
    )
    parser.add_argument(
        "--burn-value",
        type=int,
        default=1,
        help="Constant value to burn when --attribute is not set.",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=0,
        help="Background value for pixels outside polygons.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Mark all pixels touched by polygons.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Force output mask to 0/1 (removes boundary labels).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Output dtype (uint8, uint16, uint32). Default chooses based on values.",
    )
    parser.add_argument(
        "--fix-invalid",
        action="store_true",
        help="Attempt to fix invalid polygons with buffer(0).",
    )
    args = parser.parse_args()

    _require_rasterio()

    input_path = Path(args.input)
    output_path = Path(args.output)
    reference_path = Path(args.reference) if args.reference else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path, layer=args.layer)
    if gdf.empty:
        raise SystemExit("No features found in input GeoPackage.")
    gdf = _ensure_polygons(gdf)

    geoms = list(gdf.geometry)
    if args.fix_invalid:
        geoms = [_fix_invalid(geom) for geom in geoms]
    geoms = _resolve_overlaps_random(geoms, fix_invalid=args.fix_invalid)
    gdf = gdf.copy()
    gdf.geometry = geoms

    width, height, transform, crs = _load_reference_grid(
        reference_path,
        bounds=tuple(args.bounds) if args.bounds else None,
        width=args.width,
        height=args.height,
        pixel_coords=args.pixel_coords,
    )

    interior_shapes, boundary_geoms, max_value = _build_shapes(
        gdf,
        value_field=args.attribute,
        burn_value=args.burn_value,
        fix_invalid=args.fix_invalid,
        boundary_px=3.0,
    )

    boundary_value = max_value + 1
    shapes = interior_shapes + [(geom, boundary_value) for geom in boundary_geoms]
    dtype = _choose_dtype(args.dtype, boundary_value)
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=args.background,
        dtype=dtype,
        all_touched=args.all_touched,
    )
    if args.binary:
        mask = (mask > 0).astype(np.uint8)

    _write_mask(mask, output_path, transform, crs)
    print(f"Wrote mask {output_path} with shape {mask.shape}.")


if __name__ == "__main__":
    main()
