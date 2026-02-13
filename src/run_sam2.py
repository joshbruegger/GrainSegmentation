import traceback
import os
import atexit
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
import pickle
import cv2
import large_image
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm

# Handle very large images
Image.MAX_IMAGE_PIXELS = None

# Set random seed
np.random.seed(3)


DEFAULT_SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_autocast_context = None


def select_device(device_pref: Optional[str]) -> torch.device:
    pref = (device_pref or "auto").lower()
    pref = {"gpu": "cuda"}.get(pref, pref)

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_pref}")


def configure_device(device: torch.device) -> None:
    global _autocast_context
    print(f"using device: {device}")
    if device.type == "cuda":
        _autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
        _autocast_context.__enter__()
        atexit.register(lambda: _autocast_context.__exit__(None, None, None))
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def encode_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask (HxW) to simple RLE with row-major order.
    Returns a dict: { "size": [H, W], "counts": [runs...], "order": "C" }
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D HxW for RLE encoding.")
    h, w = mask.shape
    # Vectorized implementation for speed
    flat = mask.ravel(order="C") > 0
    n = len(flat)

    if n == 0:
        return {"size": [int(h), int(w)], "counts": [], "order": "C"}

    starts = np.r_[0, np.where(flat[1:] != flat[:-1])[0] + 1, n]
    lengths = np.diff(starts)

    counts = lengths.tolist()
    if flat[0]:
        counts.insert(0, 0)

    return {"size": [int(h), int(w)], "counts": [int(x) for x in counts], "order": "C"}


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
class MergeEntry:
    ann: Dict[str, Any]
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    source_ids: List[int]


def _build_merge_entries(masks: List[dict]) -> List[MergeEntry]:
    entries: List[MergeEntry] = []
    for idx, ann in enumerate(masks):
        rle = ann.get("rle")
        if not rle:
            continue

        mask = decode_rle(rle)
        if not mask.any():
            continue

        tile_x = int(ann.get("tile_x", 0))
        tile_y = int(ann.get("tile_y", 0))
        bbox = ann.get("bbox")

        crop = None
        if bbox is not None and len(bbox) == 4:
            x0, y0, w, h = (int(v) for v in bbox)
            if w > 0 and h > 0:
                lx0 = x0 - tile_x
                ly0 = y0 - tile_y
                lx1 = lx0 + w - 1
                ly1 = ly0 + h - 1
                if 0 <= lx0 <= lx1 < mask.shape[1] and 0 <= ly0 <= ly1 < mask.shape[0]:
                    crop = mask[ly0 : ly1 + 1, lx0 : lx1 + 1].copy()
                    if crop.any():
                        area = int(crop.sum())
                        entries.append(
                            MergeEntry(
                                ann=ann,
                                mask=crop,
                                bbox=(x0, y0, x0 + w - 1, y0 + h - 1),
                                area=area,
                                source_ids=[idx],
                            )
                        )
                        continue

        bbox_local = _mask_bbox(mask)
        if bbox_local is None:
            continue
        lx0, ly0, lx1, ly1 = bbox_local
        crop = mask[ly0 : ly1 + 1, lx0 : lx1 + 1].copy()
        area = int(crop.sum())
        gx0 = tile_x + lx0
        gy0 = tile_y + ly0
        gx1 = tile_x + lx1
        gy1 = tile_y + ly1
        entries.append(
            MergeEntry(
                ann=ann,
                mask=crop,
                bbox=(gx0, gy0, gx1, gy1),
                area=area,
                source_ids=[idx],
            )
        )

    return entries


def _overlap_pixels(a: MergeEntry, b: MergeEntry, min_overlap: int) -> int:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
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
    a_x0 = ox0 - ax0
    a_y0 = oy0 - ay0
    b_x0 = ox0 - bx0
    b_y0 = oy0 - by0
    a_sub = a.mask[a_y0 : a_y0 + h, a_x0 : a_x0 + w]
    b_sub = b.mask[b_y0 : b_y0 + h, b_x0 : b_x0 + w]
    return int(np.logical_and(a_sub, b_sub).sum())


def merge_overlapping_masks(
    masks: List[dict],
    min_overlap: int,
    min_iom: float,
    verbose: bool = True,
) -> List[dict]:
    if min_overlap < 1:
        raise ValueError("min_overlap must be >= 1 to represent pixel overlap")
    if not (0.0 <= min_iom <= 1.0):
        raise ValueError("min_iom must be within [0, 1].")
    entries = _build_merge_entries(masks)
    n = len(entries)
    if n == 0:
        return []

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
    iterator = tqdm(range(n), desc="Overlap merge", mininterval=60, miniters=1)
    for i in iterator:
        a = entries[i]
        for j in range(i + 1, n):
            b = entries[j]
            overlap = _overlap_pixels(a, b, min_overlap)
            if overlap >= min_overlap:
                should_merge = False
                if min_iom <= 0.0:
                    should_merge = True
                else:
                    min_area = min(a.area, b.area)
                    if min_area > 0:
                        iom = overlap / min_area
                        should_merge = iom >= min_iom
                if should_merge and find(i) != find(j):
                    union(i, j)
                    merge_pairs += 1

    groups: Dict[int, List[MergeEntry]] = {}
    for idx, entry in enumerate(entries):
        root = find(idx)
        groups.setdefault(root, []).append(entry)

    merged_masks: List[dict] = []
    for group in groups.values():
        if len(group) == 1:
            merged_masks.append(group[0].ann)
            continue

        gx0 = min(e.bbox[0] for e in group)
        gy0 = min(e.bbox[1] for e in group)
        gx1 = max(e.bbox[2] for e in group)
        gy1 = max(e.bbox[3] for e in group)
        h = gy1 - gy0 + 1
        w = gx1 - gx0 + 1
        union_mask = np.zeros((h, w), dtype=bool)

        merged_ids: List[int] = []
        for entry in group:
            merged_ids.extend(entry.source_ids)
            ex0, ey0, ex1, ey1 = entry.bbox
            y0 = ey0 - gy0
            x0 = ex0 - gx0
            union_mask[
                y0 : y0 + entry.mask.shape[0], x0 : x0 + entry.mask.shape[1]
            ] |= entry.mask

        bbox = _mask_bbox(union_mask)
        if bbox is None:
            continue
        lx0, ly0, lx1, ly1 = bbox
        if (
            lx0 > 0
            or ly0 > 0
            or lx1 < union_mask.shape[1] - 1
            or ly1 < union_mask.shape[0] - 1
        ):
            union_mask = union_mask[ly0 : ly1 + 1, lx0 : lx1 + 1]
            gx0 += lx0
            gy0 += ly0

        base = dict(group[0].ann)
        base["tile_x"] = int(gx0)
        base["tile_y"] = int(gy0)
        base["rle"] = encode_rle(union_mask.astype(np.uint8))
        base["bbox"] = [
            int(gx0),
            int(gy0),
            int(union_mask.shape[1]),
            int(union_mask.shape[0]),
        ]
        base["area"] = int(union_mask.sum())
        base["merged_ids"] = sorted(set(int(x) for x in merged_ids))
        merged_masks.append(base)

    if verbose:
        merged_groups = sum(1 for g in groups.values() if len(g) > 1)
        print(f"Overlap groups formed: {len(groups)} (merged {merge_pairs} pairs)")
        print(f"Merged masks created: {merged_groups}")

    return merged_masks


def masks_to_rle(masks: List[dict]) -> List[dict]:
    rles: List[dict] = []
    for ann in masks:
        seg = ann.get("segmentation")
        if seg is None:
            continue

        if isinstance(seg, dict) and "counts" in seg:
            rle = seg
        else:
            rle = encode_rle(seg)

        out = {"rle": rle}
        for key in (
            "area",
            "bbox",
            "predicted_iou",
            "stability_score",
            "point_coords",
            "crop_box",
            "tile_x",
            "tile_y",
        ):
            if key in ann:
                val = ann[key]
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                out[key] = val
        rles.append(out)
    return rles


def _mask_iou(a: MergeEntry, b: MergeEntry) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 > ix1 or iy0 > iy1:
        return 0.0

    a_x0 = ix0 - ax0
    a_y0 = iy0 - ay0
    a_x1 = ix1 - ax0
    a_y1 = iy1 - ay0
    b_x0 = ix0 - bx0
    b_y0 = iy0 - by0
    b_x1 = ix1 - bx0
    b_y1 = iy1 - by0

    a_slice = a.mask[a_y0 : a_y1 + 1, a_x0 : a_x1 + 1]
    b_slice = b.mask[b_y0 : b_y1 + 1, b_x0 : b_x1 + 1]
    if a_slice.size == 0 or b_slice.size == 0:
        return 0.0

    if a_slice.shape != b_slice.shape:
        h = min(a_slice.shape[0], b_slice.shape[0])
        w = min(a_slice.shape[1], b_slice.shape[1])
        if h == 0 or w == 0:
            return 0.0
        a_slice = a_slice[:h, :w]
        b_slice = b_slice[:h, :w]

    inter = int(np.logical_and(a_slice, b_slice).sum())
    if inter == 0:
        return 0.0

    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def apply_nms(masks: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    if len(masks) == 0:
        return []

    entries = _build_merge_entries(masks)
    entry_by_idx = {entry.source_ids[0]: entry for entry in entries}
    scores = np.array(
        [float(mask.get("predicted_iou", 0.0)) for mask in masks], dtype=np.float32
    )
    order = scores.argsort()[::-1].tolist()
    keep_indices: List[int] = []

    while order:
        i = order[0]
        keep_indices.append(i)
        entry_i = entry_by_idx.get(i)
        if entry_i is None:
            order = order[1:]
            continue

        remaining: List[int] = []
        for j in order[1:]:
            entry_j = entry_by_idx.get(j)
            if entry_j is None:
                remaining.append(j)
                continue
            if _mask_iou(entry_i, entry_j) <= iou_threshold:
                remaining.append(j)
        order = remaining

    return [masks[i] for i in keep_indices]


def filter_large_masks(
    masks: List[dict], tile_area: int, max_coverage: float
) -> List[dict]:
    if tile_area <= 0:
        return masks
    keep: List[dict] = []
    for mask in masks:
        area = mask.get("area")
        if area is None:
            seg = mask.get("segmentation")
            if isinstance(seg, np.ndarray):
                area = int(seg.sum())
            else:
                area = 0
        if (area / float(tile_area)) < max_coverage:
            keep.append(mask)
    return keep


def segment_image(
    image_path: str,
    output_dir: str,
    mask_generator: Optional[SAM2AutomaticMaskGenerator],
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    visualize_probability: float = 0.1,
    nms_thresh: Optional[float] = None,
    merge_overlaps: bool = False,
    merge_min_overlap: int = 1,
    merge_iom_thresh: float = 0.0,
    max_mask_coverage: Optional[float] = None,
    save_mask_cache: bool = False,
    load_mask_cache: bool = False,
    mask_cache_dir: Optional[str] = None,
):
    print(f"Segmenting image: {image_path}")
    img = large_image.open(image_path)

    if tile_size is None:
        try:
            meta = img.getMetadata()
            # Use the largest dimension to ensure the whole image is one tile
            tile_size = max(meta["sizeX"], meta["sizeY"])
            print(f"No tile size specified. Using full image size: {tile_size}")
        except Exception as e:
            print(f"Warning: Could not determine image size for default tile_size: {e}")

    # get name of file from image_path
    file_name = os.path.basename(image_path)
    file_name = os.path.splitext(file_name)[0]
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = mask_cache_dir or output_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_name}.raw_masks.pkl")

    if load_mask_cache:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Mask cache not found: {cache_path}")
        print(f"Loading mask cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_masks = pickle.load(f)
    else:
        if mask_generator is None:
            raise ValueError("mask_generator is required when not loading cache.")
        masks = []

        # Auto-calculate overlap if not specified
        current_overlap = tile_overlap
        if tile_overlap is None and tile_size is not None:
            current_overlap = int(tile_size * 0.25)
            print(
                f"Auto-selected tile overlap: {current_overlap} pixels (25% of {tile_size})"
            )

        all_masks = []

        for tile in (
            pbar := tqdm(
                img.tileIterator(
                    tile_size=dict(width=tile_size, height=tile_size),
                    tile_overlap=dict(
                        x=current_overlap, y=current_overlap, edges=False
                    ),
                    format=large_image.constants.TILE_FORMAT_NUMPY,
                ),
                mininterval=60,
                miniters=1,
            )
        ):
            # generate masks for tile
            with torch.inference_mode():
                masks = mask_generator.generate(tile["tile"])

            if max_mask_coverage is not None:
                tile_area = int(tile["tile"].shape[0] * tile["tile"].shape[1])
                before = len(masks)
                masks = filter_large_masks(masks, tile_area, max_mask_coverage)
                if before != len(masks):
                    pbar.set_description(
                        f"Filtered {before - len(masks)} large masks", refresh=False
                    )

            # current = tile["tile_position"]["position"] + 1
            # total = tile["iterator_range"]["position"]
            pbar.set_description(f"Found {len(masks)} grains", refresh=False)

            if np.random.random() < visualize_probability:
                visualize_masks(masks, tile, output_dir, file_name)

            # adjust coordinates to global
            masks = [mask_local_to_global(mask, tile) for mask in masks]

            all_masks.extend(masks_to_rle(masks))

        if save_mask_cache:
            print(f"Saving mask cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(all_masks, f, protocol=pickle.HIGHEST_PROTOCOL)

    if nms_thresh is not None:
        print(f"Applying NMS with threshold {nms_thresh} to {len(all_masks)} masks...")
        all_masks = apply_nms(all_masks, nms_thresh)
        print(f"Kept {len(all_masks)} masks after NMS.")

    if merge_overlaps:
        print(
            f"Merging overlapping masks with min overlap {merge_min_overlap} "
            f"and IoM >= {merge_iom_thresh} from {len(all_masks)} masks..."
        )
        all_masks = merge_overlapping_masks(
            all_masks, merge_min_overlap, merge_iom_thresh, verbose=True
        )
        print(f"Output {len(all_masks)} masks after overlap merge.")

    # save masks to RLE
    with open(os.path.join(output_dir, f"{file_name}.json"), "w") as f:
        json.dump(all_masks, f)
    # save masks to pickle
    # with open(os.path.join(output_dir, f"{file_name}.pkl"), "wb") as f:
    #     pickle.dump(all_masks, f)


def visualize_masks(
    masks: List[dict],
    tile: dict,
    output_dir: str,
    file_name: str,
) -> None:
    plt.imshow(tile["tile"])
    show_anns(masks)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{file_name}_tile_{tile['x']}_{tile['y']}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def mask_local_to_global(mask: dict, tile: dict) -> dict:
    mask["tile_x"] = tile["x"]
    mask["tile_y"] = tile["y"]

    # Adjust bbox [x, y, w, h] to global
    if "bbox" in mask:
        mask["bbox"][0] += tile["x"]
        mask["bbox"][1] += tile["y"]

    # Adjust point_coords to global
    if "point_coords" in mask:
        if isinstance(mask["point_coords"], (list, np.ndarray)):
            pcs = np.array(mask["point_coords"])
            pcs[:, 0] += tile["x"]
            pcs[:, 1] += tile["y"]
            mask["point_coords"] = pcs.tolist()

    # Adjust crop_box [x, y, w, h] to global
    if "crop_box" in mask:
        mask["crop_box"][0] += tile["x"]
        mask["crop_box"][1] += tile["y"]

    return mask


def collect_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images: List[str] = []
    for entry in p.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in exts:
            images.append(str(entry))
    return sorted(images)


def download_checkpoint(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return

    print(f"Checkpoint not found at {output_path}. Downloading from {url} ...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded checkpoint to {output_path}")
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Segment images using SAM 2 and export overlays + RLE."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Directory to write results to. Default: results",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_SAM2_CHECKPOINT,
        help=(
            "Path to the SAM 2 checkpoint (.pt) to load. "
            f"Default: {DEFAULT_SAM2_CHECKPOINT}"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "gpu", "mps"],
        default="auto",
        help="Device to run inference on. Default: auto (prefers GPU, falls back to CPU).",
    )
    parser.add_argument(
        "--overlay-max-side",
        type=int,
        default=1200,
        help="Max side length for saved overlay image (visualization only). Default: 1200",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for processing large images (e.g. 1024). Default: None (no tiling)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=None,
        help="Overlap between tiles in pixels. Default: None (auto-calculate 25% of tile size)",
    )
    parser.add_argument(
        "--visualize-probability",
        type=float,
        default=0.1,
        help="Probability of visualizing masks. Default: 0.1",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.5,
        help="Mask IoU threshold for NMS. Default: 0.5",
    )
    parser.add_argument(
        "--no-nms",
        action="store_true",
        help="Disable NMS post-processing.",
    )
    parser.add_argument(
        "--merge-overlap",
        action="store_true",
        help="Merge overlapping masks after NMS (if enabled).",
    )
    parser.add_argument(
        "--merge-min-overlap",
        type=int,
        default=1,
        help="Minimum overlapping pixels required to merge (default: 1).",
    )
    parser.add_argument(
        "--merge-iom-thresh",
        "--merge-iou-thresh",
        dest="merge_iom_thresh",
        type=float,
        default=0.0,
        help=(
            "Minimum IoM (intersection over minimum) required to merge overlaps "
            "(0-1). Default: 0.0. Alias: --merge-iou-thresh."
        ),
    )
    parser.add_argument(
        "--max-mask-coverage",
        type=float,
        default=None,
        help="Drop masks covering >= this fraction of a tile (0-1).",
    )
    parser.add_argument(
        "--save-mask-cache",
        action="store_true",
        help="Save raw masks to a pickle before NMS/merge.",
    )
    parser.add_argument(
        "--load-mask-cache",
        action="store_true",
        help="Load raw masks from a pickle and skip segmentation.",
    )
    parser.add_argument(
        "--mask-cache-dir",
        default=None,
        help="Directory for mask cache pickles (default: output directory).",
    )
    args = parser.parse_args()

    if args.checkpoint == DEFAULT_SAM2_CHECKPOINT and not os.path.exists(
        args.checkpoint
    ):
        download_checkpoint(SAM2_CHECKPOINT_URL, args.checkpoint)

    device = select_device(args.device)
    configure_device(device)

    if args.max_mask_coverage is not None and not (0.0 < args.max_mask_coverage <= 1.0):
        raise ValueError("--max-mask-coverage must be in (0, 1].")
    if not (0.0 <= args.merge_iom_thresh <= 1.0):
        raise ValueError("--merge-iom-thresh/--merge-iou-thresh must be in [0, 1].")

    mask_generator: Optional[SAM2AutomaticMaskGenerator] = None
    if args.load_mask_cache:
        print("Loading cached masks; skipping SAM 2 initialization.")
    else:
        sam2 = build_sam2(
            MODEL_CFG, args.checkpoint, device=device, apply_postprocessing=False
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=32,
            pred_iou_thresh=0.7,
        )

    images = collect_images(args.input)
    if len(images) == 0:
        print("No images found to process.")
        return

    print(f"Found {len(images)} image(s). Writing outputs to: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    nms_thresh = None if args.no_nms else args.nms_thresh
    merge_overlaps = args.merge_overlap

    for idx, img_path in enumerate(images, 1):
        try:
            print(f"[{idx}/{len(images)}] Processing: {img_path}")
            segment_image(
                img_path,
                args.output,
                mask_generator=mask_generator,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                visualize_probability=args.visualize_probability,
                nms_thresh=nms_thresh,
                merge_overlaps=merge_overlaps,
                merge_min_overlap=args.merge_min_overlap,
                merge_iom_thresh=args.merge_iom_thresh,
                max_mask_coverage=args.max_mask_coverage,
                save_mask_cache=args.save_mask_cache,
                load_mask_cache=args.load_mask_cache,
                mask_cache_dir=args.mask_cache_dir,
            )
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
