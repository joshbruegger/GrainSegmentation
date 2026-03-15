import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile


VALID_SUFFIXES = {".tif", ".tiff"}


def _to_channel_first_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(
            f"Expected a stacked TIFF with 3 dimensions, got shape {image.shape}."
        )

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    if image_uint8.shape[0] % 3 == 0:
        return image_uint8

    if image_uint8.shape[-1] % 3 == 0:
        return np.transpose(image_uint8, (2, 0, 1))

    raise ValueError(
        "Expected channel count divisible by 3 in either (channel, height, width) "
        f"or (height, width, channel) order, got shape {image.shape}."
    )


def split_tiff_channels(
    input_file: str | Path,
    output_dir: str | Path,
    prefix: str | None = None,
) -> list[Path]:
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.is_file():
        raise ValueError(f"Input TIFF does not exist: {input_path}")

    if input_path.suffix.lower() not in VALID_SUFFIXES:
        raise ValueError("Input file must end with .tif or .tiff")

    stacked = tifffile.imread(input_path)
    channel_first = _to_channel_first_uint8(stacked)

    if channel_first.shape[0] == 0 or channel_first.shape[0] % 3 != 0:
        raise ValueError(
            "Stacked TIFF must contain a non-empty number of channels divisible by 3."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    output_prefix = prefix or input_path.stem

    written_files = []
    for index, start in enumerate(range(0, channel_first.shape[0], 3)):
        rgb = np.transpose(channel_first[start : start + 3], (1, 2, 0))
        image_path = output_path / f"{output_prefix}_{index:03d}.tif"
        tifffile.imwrite(image_path, rgb, photometric="rgb")
        written_files.append(image_path)

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a stacked TIFF with shape (channel, height, width) or "
            "(height, width, channel) into one RGB TIFF per channel triplet."
        )
    )
    parser.add_argument("input_file", help="Input stacked TIFF path (.tif or .tiff).")
    parser.add_argument("output_dir", help="Directory for output RGB TIFF files.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. Defaults to the input TIFF stem.",
    )
    args = parser.parse_args()

    try:
        output_files = split_tiff_channels(
            args.input_file,
            args.output_dir,
            prefix=args.prefix,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved {len(output_files)} RGB TIFF files to {args.output_dir}")


if __name__ == "__main__":
    main()
