import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from collections import defaultdict

# Handle very large images
Image.MAX_IMAGE_PIXELS = None


def decode_rle(rle_dict):
    """
    Decodes a dictionary containing RLE encoded mask.
    Expected format: { "size": [H, W], "counts": [c1, c2, ...], "order": "C" }
    """
    if "size" not in rle_dict or "counts" not in rle_dict:
        raise ValueError("Invalid RLE dictionary: missing size or counts")

    h, w = rle_dict["size"]
    counts = rle_dict["counts"]

    # Heuristic to fix double-zero bug from run_sam2.py
    # If mask starts with 1, buggy encoder produced [0, 0, ...]
    # Correct encoder produces [0, x, ...] where x > 0
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


def show_anns(anns, full_shape=None, borders=True):
    if len(anns) == 0:
        return

    # Sort by area descending
    sorted_anns = sorted(anns, key=(lambda x: x.get("area", 0)), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    if full_shape:
        h, w = full_shape
    else:
        # We need the shape from the first mask
        if "rle" not in sorted_anns[0]:
            print("Error: Mask does not contain 'rle' key")
            return
        first_mask = decode_rle(sorted_anns[0]["rle"])
        h, w = first_mask.shape

    img = np.ones((h, w, 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        if "rle" not in ann:
            continue
        m = decode_rle(ann["rle"])

        tx = ann.get("tile_x", 0)
        ty = ann.get("tile_y", 0)
        mh, mw = m.shape

        # Slice bounds
        y1, y2 = ty, ty + mh
        x1, x2 = tx, tx + mw

        # Clip to image bounds
        y2 = min(y2, h)
        x2 = min(x2, w)

        if y1 >= h or x1 >= w:
            continue

        # Sub-slice of mask if clipped
        my = y2 - y1
        mx = x2 - x1
        m_sub = m[:my, :mx]

        img_slice = img[y1:y2, x1:x2]

        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img_slice[m_sub > 0] = color_mask

        if borders:
            try:
                import cv2

                contours, _ = cv2.findContours(
                    m_sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                # Try to smooth contours
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                # Draw contours on the slice
                cv2.drawContours(img_slice, contours, -1, (0, 0, 1, 0.4), thickness=1)
            except ImportError:
                pass
            except Exception:
                # cv2 might fail if mask is weird or empty
                pass

    ax.imshow(img)


def visualize_tiles(image, masks, output_prefix, borders=True):
    tile_groups = defaultdict(list)
    for ann in masks:
        tx = ann.get("tile_x", 0)
        ty = ann.get("tile_y", 0)
        tile_groups[(tx, ty)].append(ann)

    print(f"Found {len(tile_groups)} unique tiles.")

    for (tx, ty), tile_anns in tile_groups.items():
        if not tile_anns:
            continue

        # Determine tile size from the first mask in the group
        if "rle" not in tile_anns[0]:
            continue

        rle = tile_anns[0]["rle"]
        if "size" not in rle:
            first_mask = decode_rle(rle)
            h_tile, w_tile = first_mask.shape
        else:
            h_tile, w_tile = rle["size"]

        print(f"Processing tile at {tx},{ty} (size {w_tile}x{h_tile})...")

        # Crop image (lazy load)
        box = (tx, ty, tx + w_tile, ty + h_tile)
        try:
            tile_img = image.crop(box).convert("RGB")
        except Exception as e:
            print(f"Failed to crop tile {tx},{ty}: {e}")
            continue

        # Prepare anns for this tile (strip offsets)
        tile_anns_local = []
        for ann in tile_anns:
            a = ann.copy()
            a["tile_x"] = 0
            a["tile_y"] = 0
            tile_anns_local.append(a)

        # Plot
        dpi = 100
        plt.figure(figsize=(w_tile / dpi, h_tile / dpi), dpi=dpi)
        plt.imshow(tile_img)

        # Visualize masks on this tile
        show_anns(tile_anns_local, full_shape=(h_tile, w_tile), borders=borders)

        plt.axis("off")

        # Construct output filename
        base, ext = os.path.splitext(output_prefix)
        out_name = f"{base}_tile_{tx}_{ty}{ext}"

        plt.savefig(out_name, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close()
        print(f"Saved {out_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RLE masks from JSON output of run_sam2.py"
    )
    parser.add_argument("-i", "--image", required=True, help="Path to original image")
    parser.add_argument("-r", "--rle", required=True, help="Path to RLE JSON file")
    parser.add_argument(
        "-o",
        "--output",
        default="visualization.png",
        help="Output path for visualization image",
    )
    parser.add_argument(
        "--no-borders", action="store_true", help="Disable contour borders"
    )
    parser.add_argument(
        "--tiled", action="store_true", help="Visualize per-tile instead of full image"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.rle):
        print(f"Error: JSON not found: {args.rle}")
        sys.exit(1)

    print(f"Loading masks from {args.rle}...")
    with open(args.rle, "r") as f:
        masks = json.load(f)

    print(f"Loading image {args.image}...")
    try:
        # Open lazily, don't convert yet if tiled
        image = Image.open(args.image)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    if args.tiled:
        visualize_tiles(image, masks, args.output, borders=not args.no_borders)
        return

    # Full image visualization
    try:
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error converting image to RGB: {e}")
        sys.exit(1)

    h_img, w_img = image.size[1], image.size[0]

    # Setup plot
    dpi = 100
    plt.figure(figsize=(w_img / dpi, h_img / dpi), dpi=dpi)
    plt.imshow(image)

    print(f"Visualizing {len(masks)} masks...")
    show_anns(masks, full_shape=(h_img, w_img), borders=not args.no_borders)

    plt.axis("off")
    # Save with tight bounding box to remove whitespace
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()

    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
