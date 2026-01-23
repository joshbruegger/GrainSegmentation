import traceback
import os
import atexit
import urllib.request
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
import cv2
import large_image
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torchvision.ops.boxes import batched_nms
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


def apply_nms(masks: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    if len(masks) == 0:
        return []

    # Extract boxes and scores
    boxes = []
    scores = []
    for mask in masks:
        # bbox is [x, y, w, h], convert to [x1, y1, x2, y2]
        x, y, w, h = mask["bbox"]
        boxes.append([x, y, x + w, y + h])
        scores.append(mask["predicted_iou"])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    # Assume all masks are the same category (0)
    idxs = torch.zeros(len(masks), dtype=torch.int64)

    if torch.cuda.is_available():
        boxes = boxes.cuda()
        scores = scores.cuda()
        idxs = idxs.cuda()

    keep_indices = batched_nms(boxes, scores, idxs, iou_threshold)

    if keep_indices.is_cuda:
        keep_indices = keep_indices.cpu()

    return [masks[i] for i in keep_indices]


def segment_image(
    image_path: str,
    output_dir: str,
    mask_generator: SAM2AutomaticMaskGenerator,
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    visualize_probability: float = 0.1,
    nms_thresh: Optional[float] = None,
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
                tile_overlap=dict(x=current_overlap, y=current_overlap, edges=False),
                format=large_image.constants.TILE_FORMAT_NUMPY,
            ),
            mininterval=60,
            miniters=1,
        )
    ):
        # generate masks for tile
        with torch.inference_mode():
            masks = mask_generator.generate(tile["tile"])

        # current = tile["tile_position"]["position"] + 1
        # total = tile["iterator_range"]["position"]
        pbar.set_description(f"Found {len(masks)} grains", refresh=False)

        if np.random.random() < visualize_probability:
            visualize_masks(masks, tile, output_dir, file_name)

        # adjust coordinates to global
        masks = [mask_local_to_global(mask, tile) for mask in masks]

        all_masks.extend(masks_to_rle(masks))

    if nms_thresh is not None:
        print(f"Applying NMS with threshold {nms_thresh} to {len(all_masks)} masks...")
        all_masks = apply_nms(all_masks, nms_thresh)
        print(f"Kept {len(all_masks)} masks after NMS.")

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
        help="IoU threshold for NMS. Default: 0.5",
    )
    args = parser.parse_args()

    if args.checkpoint == DEFAULT_SAM2_CHECKPOINT and not os.path.exists(
        args.checkpoint
    ):
        download_checkpoint(SAM2_CHECKPOINT_URL, args.checkpoint)

    device = select_device(args.device)
    configure_device(device)

    sam2 = build_sam2(
        MODEL_CFG, args.checkpoint, device=device, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
    )

    images = collect_images(args.input)
    if len(images) == 0:
        print("No images found to process.")
        return

    print(f"Found {len(images)} image(s). Writing outputs to: {args.output}")
    os.makedirs(args.output, exist_ok=True)

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
                nms_thresh=args.nms_thresh,
            )
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
