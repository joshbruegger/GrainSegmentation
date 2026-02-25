import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np


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

    mask = np.zeros(h * w, dtype=np.uint8)
    current_idx = 0
    val = 0
    for count in counts:
        if val:
            mask[current_idx : current_idx + count] = 1
        current_idx += count
        val = 1 - val

    return mask.reshape((h, w)).astype(bool)


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


def _infer_image_extent(rle_list: List[Dict[str, Any]]) -> Tuple[int, int]:
    max_x = 0
    max_y = 0
    for ann in rle_list:
        rle = ann.get("rle")
        if not rle or "size" not in rle:
            continue
        h, w = rle["size"]
        tx = int(ann.get("tile_x", 0))
        ty = int(ann.get("tile_y", 0))
        max_x = max(max_x, tx + int(w))
        max_y = max(max_y, ty + int(h))
    return max_x, max_y


def _window_from_args(
    rle_list: List[Dict[str, Any]],
    x0: int | None,
    y0: int | None,
    width: int | None,
    height: int | None,
    bottom_right_size: int | None,
) -> Tuple[int, int, int, int]:
    if bottom_right_size is not None:
        if bottom_right_size <= 0:
            raise ValueError("--bottom-right-size must be > 0")
        max_x, max_y = _infer_image_extent(rle_list)
        x0 = max_x - bottom_right_size
        y0 = max_y - bottom_right_size
        width = bottom_right_size
        height = bottom_right_size
    if x0 is None or y0 is None or width is None or height is None:
        raise ValueError("Window requires --x0 --y0 --width --height or --bottom-right-size")
    if width <= 0 or height <= 0:
        raise ValueError("Window width/height must be > 0")
    return int(x0), int(y0), int(width), int(height)


def _intersect_box(
    ax0: int, ay0: int, ax1: int, ay1: int, bx0: int, by0: int, bx1: int, by1: int
) -> Tuple[int, int, int, int] | None:
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x0 > x1 or y0 > y1:
        return None
    return x0, y0, x1, y1


def extract_window(
    rle_list: List[Dict[str, Any]], x0: int, y0: int, width: int, height: int
) -> List[Dict[str, Any]]:
    wx1 = x0 + width - 1
    wy1 = y0 + height - 1
    out: List[Dict[str, Any]] = []
    for ann in rle_list:
        rle = ann.get("rle")
        if not rle or "size" not in rle:
            continue
        h, w = rle["size"]
        tx = int(ann.get("tile_x", 0))
        ty = int(ann.get("tile_y", 0))
        tile_x1 = tx + int(w) - 1
        tile_y1 = ty + int(h) - 1

        inter = _intersect_box(tx, ty, tile_x1, tile_y1, x0, y0, wx1, wy1)
        if inter is None:
            continue

        mask = decode_rle(rle)
        ix0, iy0, ix1, iy1 = inter
        lx0 = ix0 - tx
        ly0 = iy0 - ty
        lx1 = ix1 - tx
        ly1 = iy1 - ty
        crop = mask[ly0 : ly1 + 1, lx0 : lx1 + 1]
        if not crop.any():
            continue

        new_ann = dict(ann)
        new_ann["tile_x"] = int(ix0)
        new_ann["tile_y"] = int(iy0)
        new_ann["rle"] = encode_rle(crop)
        out.append(new_ann)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a window of RLE masks into a smaller JSON list."
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSON file with RLE list")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("--x0", type=int, help="Window top-left x (global)")
    parser.add_argument("--y0", type=int, help="Window top-left y (global)")
    parser.add_argument("--width", type=int, help="Window width")
    parser.add_argument("--height", type=int, help="Window height")
    parser.add_argument(
        "--bottom-right-size",
        type=int,
        help="Use a square window of this size at the image bottom-right",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of RLE annotations")

    x0, y0, width, height = _window_from_args(
        data, args.x0, args.y0, args.width, args.height, args.bottom_right_size
    )

    windowed = extract_window(data, x0, y0, width, height)
    with open(args.output, "w") as f:
        json.dump(windowed, f, indent=args.indent)


if __name__ == "__main__":
    main()
