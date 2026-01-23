#!/usr/bin/env python3
"""
Convert all .czi images in a folder to tiled TIFF images.

Usage:
  python src/preprocess/czi_to_tiff.py [INPUT_FOLDER]

Options:
  -o, --output-folder FOLDER   Output directory (default: shown)
  -t, --tile-size W H          Tile size, width and height (default: shown)
  -c, --compression TYPE       TIFF compression (default: shown)
  -r, --roi-file FILE          Optional text file: filename and 4 coords per line.
                                Quote filenames with spaces, e.g.:
                                "My Image #1.czi" 100 200 300 400

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
import shlex
from pathlib import Path
from typing import Iterable, Tuple
import time
import gc


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


def _filter_files(files: list[Path], only_regex: str | None) -> list[Path]:
    if not only_regex:
        return files
    pattern = re.compile(only_regex)
    filtered: list[Path] = []
    for path in files:
        name = path.name
        stem = path.stem
        if pattern.search(name) or pattern.search(stem):
            filtered.append(path)
    return filtered


def _parse_channels_arg(
    channels_arg: str | None, channel_names: list[str]
) -> list[int] | None:
    """Return list of channel indices to keep, or None to keep all.

    channels_arg accepts comma-separated values that can be integer indices or names.
    Names can match either raw names or sanitized names (case-insensitive).
    """
    if not channels_arg:
        return None
    # Build lookup maps
    name_to_idx: dict[str, int] = {}
    for idx, name in enumerate(channel_names):
        name_to_idx[name.lower()] = idx
        name_to_idx[_sanitize_filename_component(str(name)).lower()] = idx
        name_to_idx[f"c{idx}".lower()] = idx
        name_to_idx[str(idx)] = idx

    selected: list[int] = []
    for raw in channels_arg.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key in name_to_idx:
            selected.append(name_to_idx[key])
            continue
        # Try integer index
        try:
            idx = int(key)
            if 0 <= idx < len(channel_names):
                selected.append(idx)
                continue
        except Exception:
            pass
        raise RuntimeError(f"Unknown channel selector: '{raw}'. Known: {channel_names}")
    # Deduplicate preserving order
    seen: set[int] = set()
    unique = []
    for i in selected:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


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
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            parts = list(lexer)
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
    image_2d,
    out_path: Path,
    tile_size: Tuple[int, int],
    compression: str | None,
    compress_level: int | None = None,
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
    # tifffile can accept NumPy arrays and may also accept Dask arrays.
    # If a Dask array is provided, tifffile will compute as needed.
    compressionargs = None
    if compression and compress_level is not None:
        algo = str(compression).lower()
        if algo in {"deflate", "zlib", "zstd", "zstandard"}:
            compressionargs = {"level": int(compress_level)}
    kwargs = dict(
        file=str(out_path),
        data=image_2d,
        tile=tile_size,
        compression=compression,
        bigtiff=True,
        metadata=None,
    )
    if compressionargs is not None:
        kwargs["compressionargs"] = compressionargs
    tiff.imwrite(**kwargs)


def _rechunk_to_tiles_if_lazy(image_2d, tile_size: Tuple[int, int]):
    """If image_2d is a Dask array, rechunk to roughly tile-sized blocks.

    Expects tile_size as (width, height). Image is YX, so chunk as (height, width).
    """
    try:
        # Detect dask array via attribute; avoid hard import unless needed
        if hasattr(image_2d, "rechunk") and hasattr(image_2d, "chunks"):
            tile_w = int(tile_size[0])
            tile_h = int(tile_size[1])
            # image_2d is YX
            height = int(image_2d.shape[-2])
            width = int(image_2d.shape[-1])
            new_chunks = (
                max(1, min(tile_h, height)),
                max(1, min(tile_w, width)),
            )
            try:
                return image_2d.rechunk(new_chunks)
            except Exception:
                return image_2d
        return image_2d
    except Exception:
        return image_2d


def convert_folder(
    input_folder: Path,
    output_folder: str,
    tile_size: Tuple[int, int],
    compression: str | None,
    roi_map: dict[str, list[tuple[int, int, int, int]]] | None,
    verbose: bool = False,
    lazy: bool = False,
    only_regex: str | None = None,
    channels: str | None = None,
    skip_existing: bool = False,
) -> None:
    aicsimage_cls = _import_backend_aics()

    czi_files = list(_find_czi_files(input_folder))
    czi_files = _filter_files(czi_files, only_regex)
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
            t0 = time.perf_counter()
            if verbose:
                try:
                    size_bytes = src.stat().st_size
                except Exception:
                    size_bytes = -1
                print(f"[start] {src.name} (size={size_bytes} bytes)")
            img = aicsimage_cls(str(src))
            # Extract channels as CYX for first scene/time/z
            if lazy:
                try:
                    cyx = img.get_image_dask_data("CYX", S=0, T=0, Z=0)
                except Exception as e:
                    raise RuntimeError(
                        "Lazy mode requires Dask. Install: pip install 'dask[array]' 'aicspylibczi' 'tifffile>=2023.2.3' (optionally 'zarr' 'numcodecs')"
                    ) from e
            else:
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

            selected_channel_indices = _parse_channels_arg(channels, channel_names)
            if selected_channel_indices is None:
                selected_channel_indices = list(range(num_channels))

            if verbose:
                try:
                    dtype_name = str(cyx.dtype)
                except Exception:
                    dtype_name = "unknown"
                print(
                    f"[info] {src.name}: shape={tuple(cyx.shape)} dtype={dtype_name} channels={channel_names}"
                )

            # Per-image output subfolder
            per_image_dir = output_dir / src.stem
            per_image_dir.mkdir(parents=True, exist_ok=True)

            rois = _lookup_rois(roi_map, src)
            if verbose:
                print(f"[info] {src.name}: roi_count={len(rois) if rois else 0}")
            if not rois:
                for c_idx in selected_channel_indices:
                    channel_label = _sanitize_filename_component(
                        str(channel_names[c_idx])
                    )
                    dst = per_image_dir / f"{src.stem}_{channel_label}.tif"
                    image_2d = cyx[c_idx]
                    if lazy:
                        image_2d = _rechunk_to_tiles_if_lazy(image_2d, tile_size)
                    if skip_existing and dst.exists():
                        if verbose:
                            print(f"[skip] exists -> {dst.relative_to(output_dir)}")
                        print(f"[ok] {src.name} -> {dst.relative_to(output_dir)}")
                        continue
                    if verbose:
                        print(f"[write] {src.name} -> {dst.relative_to(output_dir)}")
                    _save_tiled_tiff(
                        image_2d,
                        dst,
                        tile_size,
                        compression,
                        getattr(args, "compress_level", None) if False else None,
                    )
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
                    for c_idx in selected_channel_indices:
                        channel_label = _sanitize_filename_component(
                            str(channel_names[c_idx])
                        )
                        dst = (
                            per_image_dir
                            / f"{src.stem}_{channel_label}_{roi_suffix}.tif"
                        )
                        image_2d = cyx[c_idx, yy1:yy2, xx1:xx2]
                        if lazy:
                            image_2d = _rechunk_to_tiles_if_lazy(image_2d, tile_size)
                        if skip_existing and dst.exists():
                            if verbose:
                                print(f"[skip] exists -> {dst.relative_to(output_dir)}")
                            print(
                                f"[ok] {src.name} [{roi_suffix}] -> {dst.relative_to(output_dir)}"
                            )
                            continue
                        if verbose:
                            print(
                                f"[write] {src.name} [{roi_suffix}] -> {dst.relative_to(output_dir)}"
                            )
                        _save_tiled_tiff(
                            image_2d,
                            dst,
                            tile_size,
                            compression,
                            getattr(args, "compress_level", None) if False else None,
                        )
                        print(
                            f"[ok] {src.name} [{roi_suffix}] -> {dst.relative_to(output_dir)}"
                        )
            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"[done] {src.name} in {elapsed:.1f}s")
        except Exception as e:
            print(f"[fail] {src.name}: {e}")
        finally:
            # Attempt to free resources between files
            try:
                if "img" in locals() and hasattr(img, "close"):
                    img.close()
            except Exception:
                pass
            try:
                del img
            except Exception:
                pass
            try:
                del cyx
            except Exception:
                pass
            gc.collect()


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
        help=(
            "Text file with lines: filename x1 y1 x2 y2 (x2/y2 exclusive). "
            'Quote filenames with spaces, e.g. "My Image #1.czi" 100 200 300 400'
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Regex to filter input CZI filenames (matches name or stem)",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel selectors (indices or names)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress for each file/channel/ROI",
    )
    parser.add_argument(
        "-L",
        "--lazy",
        action="store_true",
        help="Use lazy Dask-backed reads to reduce peak memory usage",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing outputs that already exist",
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
            input_folder=input_dir,
            output_folder=str(args.output_folder),
            tile_size=tile_size,
            compression=compression,
            roi_map=roi_map,
            verbose=bool(getattr(args, "verbose", False)),
            lazy=bool(getattr(args, "lazy", False)),
            only_regex=getattr(args, "only", None),
            channels=getattr(args, "channels", None),
            skip_existing=bool(getattr(args, "skip_existing", False)),
        )
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
