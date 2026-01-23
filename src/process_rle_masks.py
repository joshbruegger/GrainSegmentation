import argparse
import json
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


def _overlap_ratio(a: MaskEntry, b: MaskEntry) -> float:
    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()
    ox0 = max(ax0, bx0)
    oy0 = max(ay0, by0)
    ox1 = min(ax1, bx1)
    oy1 = min(ay1, by1)
    if ox0 > ox1 or oy0 > oy1:
        return 0.0
    ah0 = oy0 - a.tile_y
    aw0 = ox0 - a.tile_x
    bh0 = oy0 - b.tile_y
    bw0 = ox0 - b.tile_x
    h = oy1 - oy0 + 1
    w = ox1 - ox0 + 1
    a_sub = a.mask[ah0 : ah0 + h, aw0 : aw0 + w]
    b_sub = b.mask[bh0 : bh0 + h, bw0 : bw0 + w]
    overlap = int(np.logical_and(a_sub, b_sub).sum())
    min_area = max(1, min(a.area, b.area))
    return overlap / float(min_area)


def _is_contained(a: MaskEntry, b: MaskEntry) -> bool:
    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()
    if ax0 < bx0 or ay0 < by0 or ax1 > bx1 or ay1 > by1:
        return False
    a_crop, gx0, gy0 = _crop_to_bbox(a)
    by0 = gy0 - b.tile_y
    bx0 = gx0 - b.tile_x
    h, w = a_crop.shape
    if by0 < 0 or bx0 < 0:
        return False
    if by0 + h > b.mask.shape[0] or bx0 + w > b.mask.shape[1]:
        return False
    b_crop = b.mask[by0 : by0 + h, bx0 : bx0 + w]
    return bool(np.all(b_crop[a_crop]))


def _adjacent_right_coords(a: MaskEntry, b: MaskEntry) -> Tuple[np.ndarray, np.ndarray]:
    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()
    x0 = max(ax0, bx0 - 1)
    x1 = min(ax1, bx1 - 1)
    y0 = max(ay0, by0)
    y1 = min(ay1, by1)
    if x0 > x1 or y0 > y1:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    a_sub = a.mask[y0 - a.tile_y : y1 - a.tile_y + 1, x0 - a.tile_x : x1 - a.tile_x + 1]
    b_sub = b.mask[
        y0 - b.tile_y : y1 - b.tile_y + 1,
        (x0 + 1) - b.tile_x : (x1 + 1) - b.tile_x + 1,
    ]
    adj = np.logical_and(a_sub, b_sub)
    if not adj.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    ys, xs = np.where(adj)
    return xs + x0, ys + y0


def _adjacent_down_coords(a: MaskEntry, b: MaskEntry) -> Tuple[np.ndarray, np.ndarray]:
    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()
    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)
    y0 = max(ay0, by0 - 1)
    y1 = min(ay1, by1 - 1)
    if x0 > x1 or y0 > y1:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    a_sub = a.mask[y0 - a.tile_y : y1 - a.tile_y + 1, x0 - a.tile_x : x1 - a.tile_x + 1]
    b_sub = b.mask[
        (y0 + 1) - b.tile_y : (y1 + 1) - b.tile_y + 1,
        x0 - b.tile_x : x1 - b.tile_x + 1,
    ]
    adj = np.logical_and(a_sub, b_sub)
    if not adj.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    ys, xs = np.where(adj)
    return xs + x0, ys + y0


def _straight_enough(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    axis: str,
    tolerance: int,
    coverage: float,
    min_length: int,
) -> bool:
    if coords_x.size == 0:
        return False
    if axis == "vertical":
        median_x = np.median(coords_x)
        within = np.abs(coords_x - median_x) <= tolerance
        if within.mean() < coverage:
            return False
        ys = coords_y[within]
        if ys.size == 0:
            return False
        length = int(ys.max() - ys.min() + 1)
        return length >= min_length
    if axis == "horizontal":
        median_y = np.median(coords_y)
        within = np.abs(coords_y - median_y) <= tolerance
        if within.mean() < coverage:
            return False
        xs = coords_x[within]
        if xs.size == 0:
            return False
        length = int(xs.max() - xs.min() + 1)
        return length >= min_length
    return False


def _should_merge(
    a: MaskEntry,
    b: MaskEntry,
    adjacency_gap: int,
    edge_tolerance: int,
    edge_coverage: float,
    min_shared_length: int,
    max_overlap_ratio: float,
) -> bool:
    if _overlap_ratio(a, b) > max_overlap_ratio:
        return False

    ax0, ay0, ax1, ay1 = a.global_bbox()
    bx0, by0, bx1, by1 = b.global_bbox()

    x_overlap = min(ax1, bx1) - max(ax0, bx0) + 1
    y_overlap = min(ay1, by1) - max(ay0, by0) + 1

    x_gap = max(bx0 - ax1 - 1, ax0 - bx1 - 1, 0)
    y_gap = max(by0 - ay1 - 1, ay0 - by1 - 1, 0)

    candidate_vertical = y_overlap > 0 and x_gap <= adjacency_gap
    candidate_horizontal = x_overlap > 0 and y_gap <= adjacency_gap
    if not (candidate_vertical or candidate_horizontal):
        return False

    coords = []
    xs, ys = _adjacent_right_coords(a, b)
    if xs.size:
        coords.append((xs, ys))
    xs, ys = _adjacent_right_coords(b, a)
    if xs.size:
        coords.append((xs + 1, ys))

    if coords:
        xs = np.concatenate([c[0] for c in coords])
        ys = np.concatenate([c[1] for c in coords])
        if _straight_enough(xs, ys, "vertical", edge_tolerance, edge_coverage, min_shared_length):
            return True

    coords = []
    xs, ys = _adjacent_down_coords(a, b)
    if xs.size:
        coords.append((xs, ys))
    xs, ys = _adjacent_down_coords(b, a)
    if xs.size:
        coords.append((xs, ys + 1))

    if coords:
        xs = np.concatenate([c[0] for c in coords])
        ys = np.concatenate([c[1] for c in coords])
        if _straight_enough(xs, ys, "horizontal", edge_tolerance, edge_coverage, min_shared_length):
            return True

    return False


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


def _build_entries(rle_list: List[Dict[str, Any]]) -> List[MaskEntry]:
    entries: List[MaskEntry] = []
    for i, ann in enumerate(rle_list):
        rle = ann.get("rle")
        if not rle:
            continue
        mask = decode_rle(rle).astype(bool)
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        tile_x = int(ann.get("tile_x", 0))
        tile_y = int(ann.get("tile_y", 0))
        area = int(mask.sum())
        entries.append(
            MaskEntry(
                ann=ann,
                mask=mask,
                tile_x=tile_x,
                tile_y=tile_y,
                bbox=bbox,
                area=area,
                source_ids=[i],
            )
        )
    return entries


def _remove_contained(entries: List[MaskEntry], verbose: bool, progress: bool) -> List[MaskEntry]:
    sorted_entries = sorted(entries, key=lambda e: e.area)
    keep = [True] * len(sorted_entries)
    iterator = tqdm(
        enumerate(sorted_entries),
        total=len(sorted_entries),
        disable=not progress,
        desc="Contained check",
    )
    for i, a in iterator:
        if not keep[i]:
            continue
        for j in range(i + 1, len(sorted_entries)):
            if not keep[i]:
                break
            b = sorted_entries[j]
            if _is_contained(a, b):
                keep[i] = False
    if verbose:
        removed = len(sorted_entries) - sum(keep)
        print(f"Contained masks removed: {removed}")
    return [e for e, k in zip(sorted_entries, keep) if k]


def _merge_pass(
    entries: List[MaskEntry],
    adjacency_gap: int,
    edge_tolerance: int,
    edge_coverage: float,
    min_shared_length: int,
    max_overlap_ratio: float,
    verbose: bool,
    progress: bool,
) -> Tuple[List[MaskEntry], bool]:
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

    merge_count = 0
    iterator = tqdm(range(n), disable=not progress, desc="Merge pass")
    for i in iterator:
        for j in range(i + 1, n):
            if _should_merge(
                entries[i],
                entries[j],
                adjacency_gap,
                edge_tolerance,
                edge_coverage,
                min_shared_length,
                max_overlap_ratio,
            ):
                if find(i) != find(j):
                    union(i, j)
                    merge_count += 1

    if merge_count == 0:
        return entries, False

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
        print(f"Merge groups formed: {len(groups)} (merged {merge_count} pairs)")
    return merged_entries, True


def _merge_until_stable(
    entries: List[MaskEntry],
    adjacency_gap: int,
    edge_tolerance: int,
    edge_coverage: float,
    min_shared_length: int,
    max_overlap_ratio: float,
    verbose: bool,
    progress: bool,
) -> List[MaskEntry]:
    changed = True
    current = entries
    while changed:
        current, changed = _merge_pass(
            current,
            adjacency_gap,
            edge_tolerance,
            edge_coverage,
            min_shared_length,
            max_overlap_ratio,
            verbose,
            progress,
        )
    return current


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process RLE masks: remove contained masks and merge adjacent masks."
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSON file with RLE list")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--keep-contained",
        action="store_true",
        help="Do not remove masks fully contained in others",
    )
    parser.add_argument(
        "--adjacency-gap",
        type=int,
        default=1,
        help="Max pixel gap to consider masks adjacent",
    )
    parser.add_argument(
        "--edge-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance for straight shared edge detection",
    )
    parser.add_argument(
        "--edge-coverage",
        type=float,
        default=0.9,
        help="Fraction of adjacency points within straight edge band",
    )
    parser.add_argument(
        "--min-shared-length",
        type=int,
        default=8,
        help="Minimum length (pixels) of shared edge to merge",
    )
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.02,
        help="Max overlap ratio (to min area) allowed for merging",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    parser.add_argument("--verbose", action="store_true", help="Print processing stats")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of RLE annotations")

    entries = _build_entries(data)
    del data
    if args.verbose:
        print(f"Loaded masks: {len(entries)}")

    show_progress = not args.no_progress

    if not args.keep_contained:
        entries = _remove_contained(entries, args.verbose, show_progress)

    entries = _merge_until_stable(
        entries,
        adjacency_gap=args.adjacency_gap,
        edge_tolerance=args.edge_tolerance,
        edge_coverage=args.edge_coverage,
        min_shared_length=args.min_shared_length,
        max_overlap_ratio=args.max_overlap_ratio,
        verbose=args.verbose,
        progress=show_progress,
    )

    output = [e.ann for e in entries]
    with open(args.output, "w") as f:
        json.dump(output, f, indent=args.indent)

    if args.verbose:
        print(f"Output masks: {len(output)}")


if __name__ == "__main__":
    main()
