import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile


VALID_SUFFIXES = {".tif", ".tiff"}


def _discover_tiff_files(input_dir: Path, output_file: Path) -> list[Path]:
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )

    output_resolved = output_file.resolve()
    return [path for path in files if path.resolve() != output_resolved]


def _to_channel_first_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(
            f"Expected an RGB image with 3 dimensions, got shape {image.shape}."
        )

    if image.shape[-1] != 3:
        raise ValueError(
            "Expected an RGB image in (height, width, channel) order with 3 channels, "
            f"got shape {image.shape}."
        )

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return np.transpose(image_uint8, (2, 0, 1))


def stack_tiff_channels(input_dir: str | Path, output_file: str | Path) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")

    if output_path.suffix.lower() not in VALID_SUFFIXES:
        raise ValueError("Output file must end with .tif or .tiff")

    tiff_files = _discover_tiff_files(input_path, output_path)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in: {input_path}")

    stacked_images = []
    expected_hw = None

    for image_path in tiff_files:
        image = tifffile.imread(image_path)
        channel_first = _to_channel_first_uint8(image)

        if expected_hw is None:
            expected_hw = channel_first.shape[1:]
        elif channel_first.shape[1:] != expected_hw:
            raise ValueError(
                "All input TIFF images must have matching height and width. "
                f"Expected {expected_hw}, got {channel_first.shape[1:]} for {image_path}."
            )

        stacked_images.append(channel_first)

    stacked = np.concatenate(stacked_images, axis=0).astype(np.uint8, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, stacked)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stack all RGB TIFF images in a directory into one TIFF with "
            "shape (channel, height, width)."
        )
    )
    parser.add_argument("input_dir", help="Directory containing input TIFF images.")
    parser.add_argument("output_file", help="Output TIFF path (.tif or .tiff).")
    args = parser.parse_args()

    try:
        output_path = stack_tiff_channels(args.input_dir, args.output_file)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved stacked TIFF to {output_path}")


if __name__ == "__main__":
    main()
