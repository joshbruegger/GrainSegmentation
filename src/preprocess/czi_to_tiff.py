#!/usr/bin/env python3
"""
Convert all .czi images in a folder to tiled TIFF images.

Usage:
  python src/preprocess/czi_to_tiff.py [INPUT_FOLDER]

Options:
  -o, --output-folder FOLDER   Output directory (default: shown)
  -t, --tile-size W H          Tile size, width and height (default: shown)
  -c, --compression TYPE       TIFF compression (default: shown)
  -r, --roi-file FILE          Optional text file: "filename x1 y1 x2 y2" per line

If INPUT_FOLDER is omitted, the current working directory is used.

Notes:
- Exports each channel as a separate 2D YX tiled TIFF inside a per-image folder.
- If ROI file is provided, only those subregions are exported per channel. Coordinates
  are top-left (x1, y1) to bottom-right (x2, y2), with x2/y2 exclusive; out-of-bounds
  values are clipped.
"""

from __future__ import annotations

import sys
import re
import argparse
from pathlib import Path
from typing import Iterable, Tuple


def _import_backend_aics():
    """Import and return AICSImage class from aicsimageio.

    Raises a RuntimeError with an install hint if unavailable.
    """
    try:
        from aicsimageio import AICSImage  # type: ignore

        return AICSImage
    except Exception as e:
        raise RuntimeError(
            "aicsimageio is required. Install with: pip install aicsimageio"
        ) from e


def _find_czi_files(input_dir: Path) -> Iterable[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".czi"
    )


def _sanitize_filename_component(text: str) -> str:
    """Return a filesystem-safe component derived from text."""
    text = text.strip()
    if not text:
        return "unnamed"
    # Replace non-alphanumeric with underscore and collapse repeats
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "unnamed"


def _parse_roi_file(roi_file: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    mapping: dict[str, list[tuple[int, int, int, int]]] = {}
    with roi_file.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                print(
                    f"[warn] ROI line {line_no} malformed (need 5 fields): {raw.rstrip()}"
                )
                continue
            fname, sx1, sy1, sx2, sy2 = parts
            try:
                x1, y1, x2, y2 = int(sx1), int(sy1), int(sx2), int(sy2)
            except ValueError:
                print(
                    f"[warn] ROI line {line_no} has non-integer coords: {raw.rstrip()}"
                )
                continue
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x1 == x2 or y1 == y2:
                print(
                    f"[warn] ROI line {line_no} has zero-area box, skipping: {raw.rstrip()}"
                )
                continue
            key_exact = Path(fname).name.lower()
            key_stem = Path(fname).stem.lower()
            mapping.setdefault(key_exact, []).append((x1, y1, x2, y2))
            mapping.setdefault(key_stem, []).append((x1, y1, x2, y2))
    return mapping


def _lookup_rois(
    roi_map: dict[str, list[tuple[int, int, int, int]]] | None, czi_path: Path
) -> list[tuple[int, int, int, int]]:
    if not roi_map:
        return []
    key_exact = czi_path.name.lower()
    key_stem = czi_path.stem.lower()
    rois: list[tuple[int, int, int, int]] = []
    if key_exact in roi_map:
        rois.extend(roi_map[key_exact])
    if key_stem in roi_map:
        rois.extend(roi_map[key_stem])
    return rois


def _resolve_output_dir(input_dir: Path, output_folder: str) -> Path:
    out = Path(output_folder)
    return out if out.is_absolute() else (input_dir / out)


def _save_tiled_tiff(
    image_2d, out_path: Path, tile_size: Tuple[int, int], compression: str | None
):
    # Lazy import to avoid hard dependency errors on listing help
    try:
        import tifffile as tiff  # type: ignore
    except Exception as e:  # pragma: no cover - straightforward import error
        raise RuntimeError(
            "tifffile is required. Install with: pip install 'tifffile>=2023.2.3'"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use BigTIFF to be safe for large images
    tiff.imwrite(
        file=str(out_path),
        data=image_2d,
        tile=tile_size,
        compression=compression,
        bigtiff=True,
        metadata=None,
    )


def convert_folder(
    input_folder: Path,
    output_folder: str,
    tile_size: Tuple[int, int],
    compression: str | None,
    roi_map: dict[str, list[tuple[int, int, int, int]]] | None,
) -> None:
    aicsimage_cls = _import_backend_aics()

    czi_files = list(_find_czi_files(input_folder))
    if not czi_files:
        print(f"[info] No .czi files found in: {input_folder}")
        return

    output_dir = _resolve_output_dir(input_folder, output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] Converting {len(czi_files)} CZI files from '{input_folder}' -> '{output_dir}'\n"
        f"       Tile size: {tile_size}, Compression: {compression}, ROIs: {'yes' if roi_map else 'no'}"
    )

    for src in czi_files:
        try:
            img = aicsimage_cls(str(src))
            # Extract channels as CYX for first scene/time/z
            cyx = img.get_image_data("CYX", S=0, T=0, Z=0)
            if cyx.ndim != 3:
                raise RuntimeError(
                    f"Unexpected data shape for {src.name}; expected CYX with 3 dims, got {cyx.shape}"
                )
            num_channels = int(cyx.shape[0])

            # Try to gather channel names; fall back to C{index}
            try:
                channel_names = getattr(img, "channel_names", None)
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f"C{idx}" for idx in range(num_channels)]
            except Exception:
                channel_names = [f"C{idx}" for idx in range(num_channels)]

            # Per-image output subfolder
            per_image_dir = output_dir / src.stem
            per_image_dir.mkdir(parents=True, exist_ok=True)

            rois = _lookup_rois(roi_map, src)
            if not rois:
                for c_idx in range(num_channels):
                    channel_label = _sanitize_filename_component(
                        str(channel_names[c_idx])
                    )
                    dst = per_image_dir / f"{src.stem}_{channel_label}.tif"
                    image_2d = cyx[c_idx]
                    _save_tiled_tiff(image_2d, dst, tile_size, compression)
                    print(f"[ok] {src.name} -> {dst.relative_to(output_dir)}")
            else:
                height = int(cyx.shape[1])
                width = int(cyx.shape[2])
                for roi_idx, (x1, y1, x2, y2) in enumerate(rois, start=1):
                    xx1 = max(0, min(width, x1))
                    yy1 = max(0, min(height, y1))
                    xx2 = max(0, min(width, x2))
                    yy2 = max(0, min(height, y2))
                    if xx2 <= xx1 or yy2 <= yy1:
                        print(
                            f"[warn] Clipped ROI became empty for {src.name}: ({x1},{y1},{x2},{y2}), skipping"
                        )
                        continue
                    roi_suffix = f"x{xx1}_y{yy1}_x{xx2}_y{yy2}"
                    for c_idx in range(num_channels):
                        channel_label = _sanitize_filename_component(
                            str(channel_names[c_idx])
                        )
                        dst = (
                            per_image_dir
                            / f"{src.stem}_{channel_label}_{roi_suffix}.tif"
                        )
                        image_2d = cyx[c_idx, yy1:yy2, xx1:xx2]
                        _save_tiled_tiff(image_2d, dst, tile_size, compression)
                        print(
                            f"[ok] {src.name} [{roi_suffix}] -> {dst.relative_to(output_dir)}"
                        )
        except Exception as e:
            print(f"[fail] {src.name}: {e}")


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Convert CZI images to per-channel tiled TIFFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=".",
        help="Folder containing .czi files (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="tiled_tiffs",
        help="Output folder",
    )
    parser.add_argument(
        "-t",
        "--tile-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Tile size width height",
    )
    parser.add_argument(
        "-c",
        "--compression",
        default="deflate",
        help="TIFF compression e.g. deflate, lzw, zstd, none",
    )
    parser.add_argument(
        "-r",
        "--roi-file",
        default=None,
        help="Text file with lines: 'filename x1 y1 x2 y2' (x2/y2 exclusive)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv[1:])

    input_dir = Path(args.input_folder).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[error] Input folder not found or not a directory: {input_dir}")
        return 2

    try:
        # Normalize compression option: allow 'none' to mean no compression
        compression = (
            None
            if str(args.compression).lower() in {"none", "null", "false", "0"}
            else args.compression
        )
        tile_size = (int(args.tile_size[0]), int(args.tile_size[1]))
        roi_map = None
        if args.roi_file:
            roi_path = Path(args.roi_file).expanduser().resolve()
            if not roi_path.exists() or not roi_path.is_file():
                print(f"[error] ROI file not found or not a file: {roi_path}")
                return 2
            roi_map = _parse_roi_file(roi_path)
        convert_folder(
            input_dir=input_dir,
            output_folder=str(args.output_folder),
            tile_size=tile_size,
            compression=compression,
            roi_map=roi_map,
        )
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
