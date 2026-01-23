#!/usr/bin/env python3
"""
Combine folders of TIFF tiles (10x10 grid by default, no overlap) into a single
pyramidal tiled BigTIFF per folder, using libvips via pyvips for fast, low-memory
processing.

Examples
--------
Combine each immediate subfolder of an input root (default 10x10):
    python src/pipeline/combine_tiles.py /path/to/root --depth 1 --output-dir /path/to/out

Combine specific folders and let the script infer grid from file count:
    python src/pipeline/combine_tiles.py /data/tiles/A /data/tiles/B --infer-grid

Use an explicit regex for tile ordering with named groups 'row' and 'col':
    python src/pipeline/combine_tiles.py /data/tiles/A \
      --pattern '(?P<row>\d+)[-_ ]+(?P<col>\d+)\.tif$' --rows 10 --cols 10

Change compression and tile size:
    python src/pipeline/combine_tiles.py /data/tiles/A --compression zstd --tile-size 512

Requirements
------------
pip install pyvips
libvips must be available on the system.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

try:
    import pyvips  # type: ignore
except Exception:  # pragma: no cover
    print(
        "Error: pyvips is required. Install with 'pip install pyvips' and ensure libvips is available.",
        file=sys.stderr,
    )
    raise


SUPPORTED_TIFF_COMPRESSION = {
    "none",
    "jpeg",
    "deflate",
    "packbits",
    "ccittfax4",
    "lzw",
    "zstd",
}


@dataclass(frozen=True)
class TileInfo:
    path: Path
    row_index: int
    col_index: int


def natural_key(text: str) -> List[object]:
    pattern = re.compile(r"(\d+)")
    return [int(tok) if tok.isdigit() else tok for tok in pattern.split(text)]


def discover_folders(inputs: Sequence[str], depth: int) -> List[Path]:
    discovered: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {p}")
        if p.is_dir():
            if depth <= 0:
                discovered.append(p)
            else:
                # Collect folders at requested depth
                queue = [p]
                for _ in range(depth):
                    next_level: List[Path] = []
                    for q in queue:
                        next_level.extend([d for d in q.iterdir() if d.is_dir()])
                    queue = next_level
                discovered.extend(queue)
        else:
            # A single file is not a folder of tiles; treat its parent as folder
            discovered.append(p.parent)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for d in discovered:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def list_tiles(folder: Path, glob_pattern: str) -> List[Path]:
    files = sorted(folder.glob(glob_pattern))
    return [f for f in files if f.is_file()]


def build_tile_parser(
    pattern: Optional[str], zero_based: bool
) -> Callable[[Path], Optional[Tuple[int, int]]]:
    if pattern:
        regex = re.compile(pattern)

        def parse_with_regex(path: Path) -> Optional[Tuple[int, int]]:
            m = regex.search(path.name)
            if not m:
                return None
            group_keys = set(m.groupdict().keys())
            # Accept named groups: row/col or y/x
            if "row" in group_keys and "col" in group_keys:
                row_val = int(m.group("row"))
                col_val = int(m.group("col"))
            elif "y" in group_keys and "x" in group_keys:
                row_val = int(m.group("y"))
                col_val = int(m.group("x"))
            else:
                # Fallback to first two numeric groups
                nums = [int(g) for g in m.groups() if g is not None and g.isdigit()]
                if len(nums) < 2:
                    return None
                row_val, col_val = nums[0], nums[1]
            if not zero_based:
                row_val -= 1
                col_val -= 1
            return row_val, col_val

        return parse_with_regex

    # Heuristic parser: try several common patterns
    compiled = [
        re.compile(r"(?:r|row)[ _-]?(\d+).*?(?:c|col)[ _-]?(\d+)", re.IGNORECASE),
        re.compile(r"(?:y)[ _-]?(\d+).*?(?:x)[ _-]?(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)[^\d]+(\d+)"),
    ]

    def parse_heuristic(path: Path) -> Optional[Tuple[int, int]]:
        name = path.name
        for rx in compiled:
            m = rx.search(name)
            if m:
                try:
                    row_val = int(m.group(1))
                    col_val = int(m.group(2))
                except Exception:
                    continue
                # Heuristic assumes 1-based indices in filenames
                row_val -= 1
                col_val -= 1
                return row_val, col_val
        return None

    return parse_heuristic


def order_tiles(
    tiles: List[Path],
    rows: int,
    cols: int,
    parser: Callable[[Path], Optional[Tuple[int, int]]],
) -> List[Path]:
    parsed: List[Tuple[Path, Optional[Tuple[int, int]]]] = [
        (p, parser(p)) for p in tiles
    ]
    if all(rc is not None for _, rc in parsed):
        # Validate indices within grid
        seen = set()
        grid: List[Optional[Path]] = [None] * (rows * cols)
        for p, rc in parsed:
            assert rc is not None
            r, c = rc
            if r < 0 or c < 0 or r >= rows or c >= cols:
                raise ValueError(
                    f"Parsed tile index out of range for {p.name}: (row={r}, col={c}) not in 0..{rows - 1},0..{cols - 1}"
                )
            idx = r * cols + c
            if idx in seen:
                raise ValueError(
                    f"Duplicate tile index detected for {p.name} at (row={r}, col={c})"
                )
            seen.add(idx)
            grid[idx] = p
        missing = [i for i, g in enumerate(grid) if g is None]
        if missing:
            raise ValueError(
                f"Missing tiles for {len(missing)} grid positions. First few: {missing[:10]}"
            )
        return [g for g in grid if g is not None]

    # Fallback: natural sort row-major
    print(
        "Warning: Could not parse row/col from filenames consistently; falling back to natural sort."
    )
    tiles_sorted = sorted(tiles, key=lambda p: natural_key(p.name))
    if len(tiles_sorted) != rows * cols:
        raise ValueError(
            f"Tile count mismatch: have {len(tiles_sorted)} tiles but rows*cols={rows * cols}"
        )
    return tiles_sorted


def infer_grid(
    tile_count: int,
    rows: Optional[int],
    cols: Optional[int],
    prefer_square: bool,
    default_rows: int,
    default_cols: int,
) -> Tuple[int, int]:
    if rows and cols:
        if rows * cols != tile_count:
            raise ValueError(
                f"Provided rows*cols={rows * cols} does not match tile count {tile_count}"
            )
        return rows, cols
    if rows and not cols:
        if tile_count % rows != 0:
            raise ValueError(f"tile_count {tile_count} is not divisible by rows {rows}")
        return rows, tile_count // rows
    if cols and not rows:
        if tile_count % cols != 0:
            raise ValueError(f"tile_count {tile_count} is not divisible by cols {cols}")
        return tile_count // cols, cols
    # Neither provided
    if prefer_square:
        side = int(round(tile_count**0.5))
        if side * side == tile_count:
            return side, side
    # Default
    if default_rows * default_cols != tile_count:
        raise ValueError(
            f"Default rows*cols={default_rows * default_cols} does not match tile count {tile_count}. "
            f"Provide --rows/--cols or use --infer-grid."
        )
    return default_rows, default_cols


def read_image_metadata(path: Path) -> Tuple[int, int, int]:
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    return img.width, img.height, img.bands


def validate_tiles_uniformity(
    ordered_tiles: List[Path], allow_mismatch: bool
) -> Tuple[int, int, int]:
    widths: List[int] = []
    heights: List[int] = []
    bands_list: List[int] = []
    # Sampling full set is fine for 100 tiles; keep simple
    for p in ordered_tiles:
        w, h, b = read_image_metadata(p)
        widths.append(w)
        heights.append(h)
        bands_list.append(b)
    width_set = set(widths)
    height_set = set(heights)
    bands_set = set(bands_list)
    if not allow_mismatch:
        if len(width_set) != 1 or len(height_set) != 1:
            raise ValueError(
                f"Tile sizes are not uniform. Widths={sorted(width_set)}, Heights={sorted(height_set)}"
            )
        if len(bands_set) != 1:
            raise ValueError(
                f"Tile band counts are not uniform. Bands={sorted(bands_set)}"
            )
    # Choose most common as reference
    from collections import Counter

    ref_w = Counter(widths).most_common(1)[0][0]
    ref_h = Counter(heights).most_common(1)[0][0]
    ref_b = Counter(bands_list).most_common(1)[0][0]
    return ref_w, ref_h, ref_b


def mosaic_tiles(ordered_tiles: List[Path], cols: int) -> pyvips.Image:
    images = [
        pyvips.Image.new_from_file(str(p), access="sequential") for p in ordered_tiles
    ]
    # Align with shim in case of tiny mismatches
    mosaic = pyvips.Image.arrayjoin(images, across=cols, shim=True)
    return mosaic


def save_pyramidal_tiff(
    image: pyvips.Image,
    output_path: Path,
    tile_size: int,
    compression: str,
    bigtiff: bool,
    quality: int,
    predictor: Optional[str],
) -> None:
    save_kwargs = dict(
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        subifd=True,
        bigtiff=bigtiff,
        compression=compression,
    )
    if compression == "jpeg":
        save_kwargs["Q"] = quality
    if predictor and compression in {"lzw", "deflate", "zstd"}:
        save_kwargs["predictor"] = predictor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.tiffsave(str(output_path), **save_kwargs)


def process_folder(
    folder: Path,
    output_dir: Path,
    glob_pattern: str,
    rows: Optional[int],
    cols: Optional[int],
    infer: bool,
    default_rows: int,
    default_cols: int,
    parser: Callable[[Path], Optional[Tuple[int, int]]],
    tile_size: int,
    compression: str,
    bigtiff: bool,
    quality: int,
    predictor: Optional[str],
    overwrite: bool,
    allow_mismatch: bool,
    dry_run: bool,
) -> Tuple[Path, Optional[Path]]:
    tile_paths = list_tiles(folder, glob_pattern)
    if not tile_paths:
        print(
            f"No tiles found in {folder} matching pattern '{glob_pattern}'. Skipping."
        )
        return folder, None
    grid_rows, grid_cols = infer_grid(
        tile_count=len(tile_paths),
        rows=rows if not infer else None,
        cols=cols if not infer else None,
        prefer_square=infer,
        default_rows=default_rows,
        default_cols=default_cols,
    )
    ordered = order_tiles(tile_paths, grid_rows, grid_cols, parser)
    ref_w, ref_h, ref_b = validate_tiles_uniformity(ordered, allow_mismatch)
    out_name = folder.name + ".tif"
    out_path = output_dir / out_name
    if out_path.exists() and not overwrite:
        print(
            f"Output already exists for {folder}: {out_path}. Use --overwrite to replace."
        )
        return folder, out_path
    if dry_run:
        info = {
            "folder": str(folder),
            "tiles": len(ordered),
            "grid": [grid_rows, grid_cols],
            "tile_shape": [ref_h, ref_w, ref_b],
            "output": str(out_path),
            "compression": compression,
            "tile_size": tile_size,
            "bigtiff": bigtiff,
        }
        print(json.dumps(info, indent=2))
        return folder, out_path
    mosaic = mosaic_tiles(ordered, grid_cols)
    save_pyramidal_tiff(
        mosaic,
        out_path,
        tile_size=tile_size,
        compression=compression,
        bigtiff=bigtiff,
        quality=quality,
        predictor=predictor,
    )
    print(f"Wrote {out_path}")
    return folder, out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine 10x10 TIFF tiles to pyramidal BigTIFF per folder"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input folders or a root folder. Files resolve to their parent folders.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Folder depth to enumerate under each input root (0 means treat inputs as folders)",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*.tif*",
        help="Glob pattern for tile filenames (default: *.tif*)",
    )

    grid = parser.add_argument_group("Grid")
    grid.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of tile rows (0-based indices inferred from names if pattern provided)",
    )
    grid.add_argument("--cols", type=int, default=None, help="Number of tile columns")
    grid.add_argument(
        "--infer-grid",
        action="store_true",
        help="Infer grid shape (prefer square) from tile count",
    )
    grid.add_argument(
        "--default-rows",
        type=int,
        default=10,
        help="Default rows when not inferring (default: 10)",
    )
    grid.add_argument(
        "--default-cols",
        type=int,
        default=10,
        help="Default cols when not inferring (default: 10)",
    )

    ordering = parser.add_argument_group("Ordering")
    ordering.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex with named groups 'row' and 'col' (or 'y' and 'x') to parse tile indices",
    )
    ordering.add_argument(
        "--zero-based",
        action="store_true",
        help="If set with --pattern, interpret parsed indices as zero-based (default 1-based)",
    )

    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (default: alongside folder, i.e., folder's parent)",
    )
    out.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    out.add_argument(
        "--compression",
        choices=sorted(SUPPORTED_TIFF_COMPRESSION),
        default="zstd",
        help="TIFF compression (default: zstd)",
    )
    out.add_argument(
        "--tile-size", type=int, default=512, help="TIFF tile size (default: 512)"
    )
    out.add_argument(
        "--bigtiff",
        action="store_true",
        help="Write as BigTIFF (recommended for large images)",
    )
    out.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality when --compression=jpeg (default: 90)",
    )
    out.add_argument(
        "--predictor",
        choices=["none", "horizontal", "float"],
        default="horizontal",
        help="Predictor for LZW/Deflate/ZSTD (default: horizontal)",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of folders to process in parallel",
    )
    misc.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow minor size/band mismatches; pad via arrayjoin shim",
    )
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions and inferred metadata without writing outputs",
    )

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    folders = discover_folders(args.inputs, args.depth)
    if not folders:
        print("No folders discovered.")
        return 1
    parser_fn = build_tile_parser(args.pattern, zero_based=args.zero_based)
    # Resolve predictor option
    predictor = None if args.predictor == "none" else args.predictor

    # Output directory strategy: default is parent of each folder
    def output_dir_for(folder: Path) -> Path:
        return Path(args.output_dir) if args.output_dir else folder.parent

    tasks = []
    for folder in folders:
        tasks.append(
            (
                folder,
                output_dir_for(folder),
                args.glob_pattern,
                args.rows,
                args.cols,
                args.infer_grid,
                args.default_rows,
                args.default_cols,
                parser_fn,
                args.tile_size,
                args.compression,
                bool(args.bigtiff),
                args.quality,
                predictor,
                bool(args.overwrite),
                bool(args.allow_mismatch),
                bool(args.dry_run),
            )
        )

    if args.parallel <= 1:
        for t in tasks:
            process_folder(*t)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(process_folder, *t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"Error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
