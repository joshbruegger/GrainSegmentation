import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile

Image.MAX_IMAGE_PIXELS = None


def _parse_bbox(value: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("BBox must be minx,miny,maxx,maxy")
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BBox values must be numeric") from exc
    if minx > maxx:
        minx, maxx = maxx, minx
    if miny > maxy:
        miny, maxy = maxy, miny
    return minx, miny, maxx, maxy


def _parse_channels(value: str) -> List[int]:
    if not value:
        return list(range(7))
    items: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2:
                raise argparse.ArgumentTypeError(f"Invalid channel range: {token}")
            start, end = (int(x.strip()) for x in bounds)
            if start <= end:
                items.extend(range(start, end + 1))
            else:
                items.extend(range(start, end - 1, -1))
        else:
            items.append(int(token))
    if not items:
        raise argparse.ArgumentTypeError("Channels list is empty")
    return sorted(set(items))


def _dedupe_consecutive(
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not points:
        return []
    out = [points[0]]
    for pt in points[1:]:
        if pt != out[-1]:
            out.append(pt)
    return out


def _clip_edge(
    points: Sequence[Tuple[float, float]],
    edge: str,
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = bbox

    def inside(pt: Tuple[float, float]) -> bool:
        x, y = pt
        if edge == "left":
            return x >= minx
        if edge == "right":
            return x <= maxx
        if edge == "bottom":
            return y >= miny
        if edge == "top":
            return y <= maxy
        raise ValueError(f"Unknown edge: {edge}")

    def intersect(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Tuple[float, float] | None:
        x1, y1 = p1
        x2, y2 = p2
        if edge in ("left", "right"):
            x_edge = minx if edge == "left" else maxx
            if x2 == x1:
                return None
            t = (x_edge - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x_edge, y)
        y_edge = miny if edge == "bottom" else maxy
        if y2 == y1:
            return None
        t = (y_edge - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return (x, y_edge)

    if not points:
        return []

    output: List[Tuple[float, float]] = []
    prev = points[-1]
    prev_inside = inside(prev)
    for curr in points:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                inter = intersect(prev, curr)
                if inter is not None:
                    output.append(inter)
            output.append(curr)
        elif prev_inside:
            inter = intersect(prev, curr)
            if inter is not None:
                output.append(inter)
        prev = curr
        prev_inside = curr_inside
    return output


def _clip_ring(
    ring: Sequence[Sequence[float]],
    bbox: Tuple[float, float, float, float],
) -> List[List[float]]:
    if len(ring) < 3:
        return []
    pts = [(float(x), float(y)) for x, y in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    for edge in ("left", "right", "bottom", "top"):
        pts = _clip_edge(pts, edge, bbox)
        if not pts:
            return []
    pts = _dedupe_consecutive(pts)
    if len(pts) < 3:
        return []
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return [[float(x), float(y)] for x, y in pts]


def _clip_polygon_rings(
    rings: Sequence[Sequence[Sequence[float]]],
    bbox: Tuple[float, float, float, float],
) -> List[List[List[float]]] | None:
    if not rings:
        return None
    outer = _clip_ring(rings[0], bbox)
    if not outer or len(outer) < 4:
        return None
    clipped: List[List[List[float]]] = [outer]
    for hole in rings[1:]:
        hole_clip = _clip_ring(hole, bbox)
        if hole_clip and len(hole_clip) >= 4:
            clipped.append(hole_clip)
    return clipped


def _clip_geometry(
    geom: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any] | None:
    geom_type = geom.get("type")
    coords = geom.get("coordinates")
    if geom_type == "Polygon" and isinstance(coords, list):
        rings = _clip_polygon_rings(coords, bbox)
        if rings is None:
            return None
        return {"type": "Polygon", "coordinates": rings}
    if geom_type == "MultiPolygon" and isinstance(coords, list):
        out_polys: List[List[List[List[float]]]] = []
        for poly in coords:
            if not isinstance(poly, list):
                continue
            rings = _clip_polygon_rings(poly, bbox)
            if rings is None:
                continue
            out_polys.append(rings)
        if not out_polys:
            return None
        return {"type": "MultiPolygon", "coordinates": out_polys}
    return None


def _clip_geojson(
    data: Dict[str, Any], bbox: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON features must be a list")

    clipped_features = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        clipped_geom = _clip_geometry(geom, bbox)
        if clipped_geom is None:
            continue
        new_feature = dict(feature)
        new_feature["geometry"] = clipped_geom
        clipped_features.append(new_feature)

    out = dict(data)
    out["features"] = clipped_features
    return out


def _shift_polygon_rings(
    rings: List[List[List[float]]], dx: float, dy: float
) -> List[List[List[float]]]:
    return [[[x + dx, y + dy] for x, y in ring] for ring in rings]


def _shift_geometry(geom: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    geom_type = geom.get("type")
    coords = geom.get("coordinates")
    if geom_type == "Polygon" and isinstance(coords, list):
        return {"type": "Polygon", "coordinates": _shift_polygon_rings(coords, dx, dy)}
    if geom_type == "MultiPolygon" and isinstance(coords, list):
        return {
            "type": "MultiPolygon",
            "coordinates": [_shift_polygon_rings(poly, dx, dy) for poly in coords],
        }
    return geom


def _shift_geojson(data: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    features = data.get("features", [])
    shifted_features = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        new_feature = dict(feature)
        new_feature["geometry"] = _shift_geometry(geom, dx, dy)
        shifted_features.append(new_feature)
    out = dict(data)
    out["features"] = shifted_features
    return out


def _bbox_geojson_to_pixel(
    bbox: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bbox
    left = int(round(minx))
    right = int(round(maxx))
    top = int(round(-maxy))
    bottom = int(round(-miny))
    return left, top, right, bottom


def _infer_layout(shape: Tuple[int, ...]) -> Tuple[int, int, str]:
    if len(shape) == 2:
        return shape[0], shape[1], "hw"
    if len(shape) == 3:
        if shape[0] <= 8 and shape[1] > 64 and shape[2] > 64:
            return shape[1], shape[2], "chw"
        if shape[2] <= 8 and shape[0] > 64 and shape[1] > 64:
            return shape[0], shape[1], "hwc"
        return shape[0], shape[1], "hwc"
    raise ValueError(f"Unsupported image shape: {shape}")


def _slice_array(
    arr: Any,
    layout: str,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> np.ndarray:
    if layout == "hw":
        return np.asarray(arr[crop_top:crop_bottom, crop_left:crop_right])
    if layout == "hwc":
        return np.asarray(arr[crop_top:crop_bottom, crop_left:crop_right, :])
    if layout == "chw":
        return np.asarray(arr[:, crop_top:crop_bottom, crop_left:crop_right])
    raise ValueError(f"Unsupported layout: {layout}")


def _read_tiff_crop(
    input_path: str,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        height, width, layout = _infer_layout(series.shape)
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(width, crop_right)
        crop_bottom = min(height, crop_bottom)
        if crop_left >= crop_right or crop_top >= crop_bottom:
            raise ValueError("Crop outside image bounds")
        try:
            store = series.aszarr()
            try:
                import zarr  # type: ignore
            except ImportError as exc:
                raise RuntimeError("zarr is required for tiled TIFF reading") from exc
            z = zarr.open(store, mode="r")
            cropped = _slice_array(
                z, layout, crop_left, crop_top, crop_right, crop_bottom
            )
        except Exception:
            arr = series.asarray()
            cropped = _slice_array(
                arr, layout, crop_left, crop_top, crop_right, crop_bottom
            )
    return cropped, (crop_left, crop_top, crop_right, crop_bottom)


def _crop_images(
    image_dir: str,
    sample: str,
    channels: Iterable[int],
    template: str,
    out_dir: str,
    suffix: str,
    pixel_window: Tuple[int, int, int, int],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    left, top, right, bottom = pixel_window

    for channel in channels:
        filename = template.format(sample=sample, channel=channel)
        input_path = os.path.join(image_dir, filename)
        if not os.path.exists(input_path):
            print(f"Warning: missing image {input_path}", file=sys.stderr)
            continue

        stem = Path(filename).stem
        ext = Path(filename).suffix
        output_name = f"{stem}{suffix}{ext}"
        output_path = os.path.join(out_dir, output_name)

        try:
            (
                cropped,
                (
                    crop_left,
                    crop_top,
                    crop_right,
                    crop_bottom,
                ),
            ) = _read_tiff_crop(input_path, left, top, right, bottom)
            tifffile.imwrite(output_path, cropped)
        except Exception as exc:
            print(
                f"Warning: TIFF read failed for {input_path}: {exc}",
                file=sys.stderr,
            )
            with Image.open(input_path) as img:
                width, height = img.size
                crop_left = max(0, left)
                crop_top = max(0, top)
                crop_right = min(width, right)
                crop_bottom = min(height, bottom)
                if crop_left >= crop_right or crop_top >= crop_bottom:
                    print(
                        f"Warning: crop outside image for {input_path}",
                        file=sys.stderr,
                    )
                    continue
                cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
                cropped_img.save(output_path)

        print(
            f"Saved {output_path} ({crop_right - crop_left}x{crop_bottom - crop_top})"
        )


def main() -> None:
    example = (
        "Example:\n"
        "  python src/preprocess/crop_geojson_images.py \\\n"
        "    --geojson /scratch/s4361687/GrainSeg/dataset/MWD-1#121/tempData.geojson \\\n"
        "    --out-geojson /scratch/s4361687/GrainSeg/dataset/MWD-1#121/tempData_crop.geojson \\\n"
        "    --image-dir /scratch/s4361687/GrainSeg/dataset \\\n"
        "    --sample MWD-1#121\n"
    )
    parser = argparse.ArgumentParser(
        description="Crop GeoJSON polygons and s0c0..s0c6 TIFF images to a bbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example,
    )
    parser.add_argument(
        "--geojson",
        "--vector",
        dest="geojson",
        help="Input vector file path (GeoJSON or GPKG)",
    )
    parser.add_argument(
        "--out-geojson",
        "--out-vector",
        dest="out_geojson",
        help="Output vector file path (GeoJSON or GPKG)",
    )
    parser.add_argument(
        "--bbox",
        type=_parse_bbox,
        default="0,-5000,35000,0",
        help="GeoJSON bbox minx,miny,maxx,maxy",
    )
    parser.add_argument("--image-dir", help="Directory of input TIFFs")
    parser.add_argument("--sample", help="Sample name, e.g. MWD-1#121")
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default="0-6",
        help="Channels list, e.g. 0-6 or 0,1,2",
    )
    parser.add_argument(
        "--image-template",
        default="{sample}_s0c{channel}.tif",
        help="Filename template for images",
    )
    parser.add_argument(
        "--out-image-dir",
        default=None,
        help="Output directory for cropped images (default: <image-dir>/cropped)",
    )
    parser.add_argument(
        "--suffix", default="_crop", help="Suffix for cropped image filenames"
    )
    args = parser.parse_args()

    if not (args.geojson and args.out_geojson) and not (args.image_dir and args.sample):
        parser.error(
            "Must provide either vector arguments (--geojson, --out-geojson) or "
            "image arguments (--image-dir, --sample), or both."
        )

    if bool(args.geojson) != bool(args.out_geojson):
        parser.error("Both --geojson and --out-geojson must be provided together.")

    if bool(args.image_dir) != bool(args.sample):
        parser.error("Both --image-dir and --sample must be provided together.")

    bbox = args.bbox

    if args.geojson and args.out_geojson:
        in_path = args.geojson
        out_path = args.out_geojson

        is_gpkg = in_path.endswith(".gpkg") or out_path.endswith(".gpkg")

        if is_gpkg:
            import geopandas as gpd
            from shapely.geometry import box

            print(f"Reading vector file {in_path} with geopandas...")
            gdf = gpd.read_file(in_path)
            minx, miny, maxx, maxy = bbox
            clip_box = box(minx, miny, maxx, maxy)

            print(f"Clipping to bbox {bbox}...")
            clipped_gdf = gpd.clip(gdf, clip_box)

            print(f"Shifting coordinates by x={-minx}, y={-maxy}...")
            clipped_gdf.geometry = clipped_gdf.geometry.translate(
                xoff=-minx, yoff=-maxy
            )

            driver = "GPKG" if out_path.endswith(".gpkg") else "GeoJSON"
            clipped_gdf.to_file(out_path, driver=driver)
            print(f"Wrote vector data: {out_path}")
        else:
            with open(in_path, "r") as f:
                data = json.load(f)
            minx, miny, maxx, maxy = bbox
            clipped = _clip_geojson(data, bbox)
            print(f"Shifting coordinates by x={-minx}, y={-maxy}...")
            shifted = _shift_geojson(clipped, -minx, -maxy)
            with open(out_path, "w") as f:
                json.dump(shifted, f, indent=2)
            print(f"Wrote GeoJSON: {out_path}")

    if args.image_dir and args.sample:
        out_image_dir = args.out_image_dir or os.path.join(args.image_dir, "cropped")
        pixel_window = _bbox_geojson_to_pixel(bbox)
        _crop_images(
            image_dir=args.image_dir,
            sample=args.sample,
            channels=args.channels,
            template=args.image_template,
            out_dir=out_image_dir,
            suffix=args.suffix,
            pixel_window=pixel_window,
        )


if __name__ == "__main__":
    main()
