import json
import argparse
import numpy as np
import cv2
import sys
from typing import List, Dict, Any


def decode_rle(rle_dict: Dict[str, Any]) -> np.ndarray:
    """
    Decodes a dictionary containing RLE encoded mask.
    Expected format: { "size": [H, W], "counts": [c1, c2, ...], "order": "C" }
    """
    if "size" not in rle_dict or "counts" not in rle_dict:
        raise ValueError("Invalid RLE dictionary: missing size or counts")

    h, w = rle_dict["size"]
    counts = rle_dict["counts"]

    # Heuristic to fix double-zero bug from run_sam2.py
    if len(counts) >= 2 and counts[0] == 0 and counts[1] == 0:
        counts = counts[1:]

    # Decode
    mask = np.zeros(h * w, dtype=np.uint8)
    current_idx = 0
    val = 0
    for count in counts:
        if val:
            mask[current_idx : current_idx + count] = 1
        current_idx += count
        val = 1 - val

    return mask.reshape((h, w))


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask (HxW) to simple RLE with row-major order.
    Returns a dict: { "size": [H, W], "counts": [runs...], "order": "C" }
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D HxW for RLE encoding.")
    h, w = mask.shape
    flat = (mask.astype(np.uint8).ravel(order="C") > 0).astype(np.uint8)
    counts: List[int] = []
    # RLE convention: start with run of zeros
    run_val = 0
    run_len = 0
    for v in flat:
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = v
            run_len = 1
    counts.append(run_len)
    return {"size": [int(h), int(w)], "counts": [int(x) for x in counts], "order": "C"}


def mask_to_polygons(
    mask: np.ndarray,
    offset_x: int = 0,
    offset_y: int = 0,
) -> List[List[List[float]]]:
    """
    Convert a binary mask to a list of polygons (contours).
    Returns a list of polygons, where each polygon is a list of [x, y] coordinates.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Simplify contour slightly to reduce points
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Convert to simple list of points and apply offset
        points = []
        for point in approx:
            x, y = point[0]
            gx = float(x + offset_x)
            gy = float(y + offset_y)
            gy = float(0 - gy)
            points.append([gx, gy])

        # Close the polygon
        if points[0] != points[-1]:
            points.append(points[0])

        polygons.append(points)

    return polygons


def _infer_full_height(rle_data: List[Dict[str, Any]]) -> int | None:
    max_height = 0
    found = False
    for ann in rle_data:
        rle = ann.get("rle")
        if not rle or "size" not in rle:
            continue
        try:
            h = int(rle["size"][0])
        except Exception:
            continue
        ty = int(ann.get("tile_y", 0))
        max_height = max(max_height, ty + h)
        found = True
    return max_height if found and max_height > 0 else None


def rle_to_geojson(
    rle_data: List[Dict[str, Any]],
    flip_y: bool = True,
) -> Dict[str, Any]:
    features = []
    full_height = None
    if flip_y:
        full_height = 0

    for i, ann in enumerate(rle_data):
        if "rle" not in ann:
            continue

        try:
            mask = decode_rle(ann["rle"])
        except Exception as e:
            print(f"Warning: Failed to decode RLE for item {i}: {e}")
            continue

        tx = ann.get("tile_x", 0)
        ty = ann.get("tile_y", 0)

        polys = mask_to_polygons(mask, tx, ty)

        for poly in polys:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        poly
                    ],  # GeoJSON Polygon is list of rings (outer, inner...)
                },
                "properties": {
                    "id": i,
                    **{
                        k: v for k, v in ann.items() if k not in ["rle", "segmentation"]
                    },
                },
            }
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def geojson_to_rle(
    geojson_data: Dict[str, Any], height: int, width: int
) -> List[Dict[str, Any]]:
    rle_list = []

    # We will create one mask per feature
    # Note: If features overlap, this creates separate masks.

    for i, feature in enumerate(geojson_data.get("features", [])):
        geom = feature.get("geometry")
        if not geom or geom["type"] != "Polygon":
            # Support MultiPolygon? For now, stick to Polygon
            if geom["type"] == "MultiPolygon":
                # Handle MultiPolygon by splitting or rendering all
                pass  # TODO
            continue

        # Parse coordinates
        # GeoJSON Polygon: [ [ [x,y], ... ], [ [x,y], ... ] ] (First is outer, others are holes)
        rings = geom["coordinates"]
        if not rings:
            continue

        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw outer ring
        outer_ring = np.array(rings[0], dtype=np.int32)
        cv2.fillPoly(mask, [outer_ring], 1)

        # Draw holes (subtract)
        for hole in rings[1:]:
            hole_ring = np.array(hole, dtype=np.int32)
            cv2.fillPoly(mask, [hole_ring], 0)

        # Encode
        rle = encode_rle(mask)

        ann = {"rle": rle, "tile_x": 0, "tile_y": 0, **feature.get("properties", {})}
        rle_list.append(ann)

    return rle_list


def main():
    parser = argparse.ArgumentParser(
        description="Convert between RLE masks and GeoJSON polygons."
    )
    parser.add_argument(
        "mode", choices=["rle2json", "json2rle"], help="Conversion mode"
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "--height", type=int, help="Image height (required for json2rle)"
    )
    parser.add_argument("--width", type=int, help="Image width (required for json2rle)")
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Do not flip Y when converting RLE to GeoJSON",
    )

    args = parser.parse_args()

    if args.mode == "rle2json":
        print(f"Reading RLE data from {args.input}...")
        with open(args.input, "r") as f:
            data = json.load(f)

        print("Converting to GeoJSON...")
        geojson = rle_to_geojson(
            data,
            flip_y=not args.no_flip_y,
        )

        print(f"Writing to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(geojson, f, indent=2)

    elif args.mode == "json2rle":
        if not args.height or not args.width:
            print("Error: --height and --width are required for json2rle conversion")
            sys.exit(1)

        print(f"Reading GeoJSON from {args.input}...")
        with open(args.input, "r") as f:
            data = json.load(f)

        print("Converting to RLE...")
        rle_data = geojson_to_rle(data, args.height, args.width)

        print(f"Writing to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(rle_data, f)

    print("Done.")


if __name__ == "__main__":
    main()
