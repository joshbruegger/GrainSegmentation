import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


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


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())
    return x0, y0, x1, y1


@dataclass
class MaskEntry:
    ann: Dict[str, Any]
    mask: np.ndarray
    tile_x: int
    tile_y: int
    bbox: Tuple[int, int, int, int]
    area: int
    source_ids: List[int]

    def global_bbox(self) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.bbox
        return x0 + self.tile_x, y0 + self.tile_y, x1 + self.tile_x, y1 + self.tile_y


def _crop_to_bbox(entry: MaskEntry) -> Tuple[np.ndarray, int, int]:
    x0, y0, x1, y1 = entry.bbox
    crop = entry.mask[y0 : y1 + 1, x0 : x1 + 1]
    gx0 = entry.tile_x + x0
    gy0 = entry.tile_y + y0
    return crop, gx0, gy0


def _overlap_pixels(a: MaskEntry, b: MaskEntry, min_overlap: int) -> int:
    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()
    ox0 = max(ax0, bx0)
    oy0 = max(ay0, by0)
    ox1 = min(ax1, bx1)
    oy1 = min(ay1, by1)
    if ox0 > ox1 or oy0 > oy1:
        return 0
    h = oy1 - oy0 + 1
    w = ox1 - ox0 + 1
    if h * w < min_overlap:
        return 0
    ah0 = oy0 - a.tile_y
    aw0 = ox0 - a.tile_x
    bh0 = oy0 - b.tile_y
    bw0 = ox0 - b.tile_x
    a_sub = a.mask[ah0 : ah0 + h, aw0 : aw0 + w]
    b_sub = b.mask[bh0 : bh0 + h, bw0 : bw0 + w]
    return int(np.logical_and(a_sub, b_sub).sum())


def _merge_group(entries: List[MaskEntry]) -> MaskEntry:
    gx0 = min(e.global_bbox()[0] for e in entries)
    gy0 = min(e.global_bbox()[1] for e in entries)
    gx1 = max(e.global_bbox()[2] for e in entries)
    gy1 = max(e.global_bbox()[3] for e in entries)
    h = gy1 - gy0 + 1
    w = gx1 - gx0 + 1
    union_mask = np.zeros((h, w), dtype=bool)

    merged_ids: List[int] = []
    for entry in entries:
        merged_ids.extend(entry.source_ids)
        crop, cx0, cy0 = _crop_to_bbox(entry)
        y0 = cy0 - gy0
        x0 = cx0 - gx0
        union_mask[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] |= crop

    base = dict(entries[0].ann)
    base["tile_x"] = int(gx0)
    base["tile_y"] = int(gy0)
    base["rle"] = encode_rle(union_mask.astype(np.uint8))
    base["merged_ids"] = sorted(set(int(x) for x in merged_ids))

    bbox = _mask_bbox(union_mask)
    if bbox is None:
        raise ValueError("Merged mask became empty")
    area = int(union_mask.sum())
    return MaskEntry(
        ann=base,
        mask=union_mask,
        tile_x=int(gx0),
        tile_y=int(gy0),
        bbox=bbox,
        area=area,
        source_ids=base["merged_ids"],
    )


def _build_entries(rle_list: List[Dict[str, Any]], progress: bool) -> List[MaskEntry]:
    entries: List[MaskEntry] = []
    iterator = tqdm(
        rle_list,
        total=len(rle_list),
        disable=not progress,
        desc="Decode masks",
        mininterval=60,
        miniters=1,
    )
    for i, ann in enumerate(iterator):
        rle = ann.get("rle")
        if not rle:
            continue
        mask = decode_rle(rle).astype(bool)
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        tile_x = int(ann.get("tile_x", 0))
        tile_y = int(ann.get("tile_y", 0))
        x0, y0, x1, y1 = bbox
        crop = mask[y0 : y1 + 1, x0 : x1 + 1].copy()
        if not crop.any():
            continue
        tile_x += x0
        tile_y += y0
        bbox = (0, 0, crop.shape[1] - 1, crop.shape[0] - 1)
        area = int(crop.sum())
        entries.append(
            MaskEntry(
                ann=ann,
                mask=crop,
                tile_x=tile_x,
                tile_y=tile_y,
                bbox=bbox,
                area=area,
                source_ids=[i],
            )
        )
    return entries


def _merge_overlapping(
    entries: List[MaskEntry],
    min_overlap: int,
    min_iou: float,
    verbose: bool,
    progress: bool,
) -> List[MaskEntry]:
    n = len(entries)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    merge_pairs = 0
    iterator = tqdm(range(n), disable=not progress, desc="Overlap merge")
    for i in iterator:
        for j in range(i + 1, n):
            overlap = _overlap_pixels(entries[i], entries[j], min_overlap)
            if overlap >= min_overlap:
                should_merge = False
                if min_iou <= 0.0:
                    should_merge = True
                else:
                    union_area = entries[i].area + entries[j].area - overlap
                    if union_area > 0:
                        iou = overlap / union_area
                        should_merge = iou >= min_iou
                if should_merge and find(i) != find(j):
                    union(i, j)
                    merge_pairs += 1

    groups: Dict[int, List[MaskEntry]] = {}
    for idx, entry in enumerate(entries):
        root = find(idx)
        groups.setdefault(root, []).append(entry)

    merged_entries = []
    for group in groups.values():
        if len(group) == 1:
            merged_entries.append(group[0])
        else:
            merged_entries.append(_merge_group(group))

    if verbose:
        merged_masks = sum(1 for g in groups.values() if len(g) > 1)
        print(f"Overlap groups formed: {len(groups)} (merged {merge_pairs} pairs)")
        print(f"Merged masks created: {merged_masks}")

    return merged_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge any RLE masks that overlap by a configurable amount."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input JSON file with RLE list"
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum overlapping pixels required to merge (default: 1)",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        help="Minimum IoU required to merge (0-1). Default: 0.0",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    parser.add_argument("--verbose", action="store_true", help="Print processing stats")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Optional path to cache decoded entries for faster re-runs. "
            "If the cache exists, it is used instead of re-reading JSON."
        ),
    )
    args = parser.parse_args()

    if args.min_overlap < 1:
        raise ValueError("--min-overlap must be >= 1 to represent pixel overlap")
    if not (0.0 <= args.min_iou <= 1.0):
        raise ValueError("--min-iou must be in [0, 1]")

    show_progress = not args.no_progress
    entries: List[MaskEntry]
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached entries from {args.cache}...", flush=True)
        start = time.time()
        with open(args.cache, "rb") as f:
            entries = pickle.load(f)
        if args.verbose:
            print(f"Loaded cache in {time.time() - start:.1f}s")
    else:
        size_gb = os.path.getsize(args.input) / (1024**3)
        print(f"Loading JSON input ({size_gb:.2f} GB)...", flush=True)
        start = time.time()
        with open(args.input, "r") as f:
            data = json.load(f)
        if args.verbose:
            print(f"Parsed JSON in {time.time() - start:.1f}s")

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of RLE annotations")

        entries = _build_entries(data, show_progress)
        del data

        if args.cache:
            print(f"Writing cache to {args.cache}...", flush=True)
            with open(args.cache, "wb") as f:
                pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.verbose:
        print(f"Loaded masks: {len(entries)}")

    entries = _merge_overlapping(
        entries, args.min_overlap, args.min_iou, args.verbose, show_progress
    )

    output = [e.ann for e in entries]
    with open(args.output, "w") as f:
        json.dump(output, f, indent=args.indent)

    if args.verbose:
        print(f"Output masks: {len(output)}")


if __name__ == "__main__":
    main()
