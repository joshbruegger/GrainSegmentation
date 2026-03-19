"""
Evaluate trained YOLO segmentation models: Ultralytics val, export benchmark, or SAHI tiled prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tifffile import TiffFile

from config import variant_choices
from pipeline import resolve_variant_paths
from train import _parse_device
from visualize_dataset import collect_samples, resolve_split_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 segmentation evaluation: val, benchmark, or SAHI sliced inference."
    )
    parser.add_argument(
        "--mode",
        choices=("val", "benchmark", "sahi"),
        required=True,
        help="val: Ultralytics validator; benchmark: export/runtime benchmark; sahi: tiled prediction on val images.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained weights (.pt), e.g. best.pt or last.pt.",
    )
    parser.add_argument(
        "--variant",
        choices=variant_choices(),
        default=None,
        help="Dataset variant (resolves dataset YAML under $SCRATCH/GrainSeg/...).",
    )
    parser.add_argument(
        "--data",
        default=None,
        type=Path,
        help="Explicit path to dataset YAML (overrides --variant).",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device: 0, 0,1, cpu, etc.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory for val artifacts (Ultralytics).",
    )
    parser.add_argument(
        "--name",
        default="eval",
        help="Run name under project for val outputs.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FP16 for val/benchmark (when supported).",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validation plots (val mode).",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save predictions JSON (val mode).",
    )
    # benchmark
    parser.add_argument(
        "--format",
        default="",
        help="Export format for benchmark only (e.g. onnx). Empty = all formats.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verbose benchmark logging.",
    )
    # sahi
    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1024)
    parser.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.2,
        help="Slice overlap ratio (SAHI).",
    )
    parser.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.2,
        help="Slice overlap ratio (SAHI).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for SAHI AutoDetectionModel.",
    )
    parser.add_argument(
        "--sahi-out-dir",
        type=Path,
        default=None,
        help="Directory for SAHI prediction_visual.png per image.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit SAHI val images processed (debug/smoke).",
    )

    args = parser.parse_args(argv)
    if not args.variant and not args.data:
        parser.error("one of --variant or --data is required")
    if args.mode == "sahi" and args.sahi_out_dir is None:
        parser.error("--sahi-out-dir is required for --mode sahi")
    if args.slice_height <= 0 or args.slice_width <= 0:
        parser.error("slice dimensions must be positive")
    return args


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.data is not None:
        path = args.data.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path}")
        return path
    resolved = resolve_variant_paths(variant_name=args.variant)
    if not resolved.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {resolved.data_yaml}")
    return resolved.data_yaml


def load_dataset_config_from_yaml(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    with data_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    return dataset_root, config


def device_for_sahi(device: int | str | list[int]) -> str:
    if device == "cpu" or device == -1:
        return "cpu"
    if isinstance(device, list):
        if not device:
            return "cpu"
        return f"cuda:{device[0]}"
    if isinstance(device, int):
        if device < 0:
            return "cpu"
        return f"cuda:{device}"
    if isinstance(device, str):
        if device == "cpu":
            return "cpu"
        if "," in device:
            first = device.split(",")[0].strip()
            return f"cuda:{first}" if first.lstrip("-").isdigit() else device
        if device.lstrip("-").isdigit():
            return f"cuda:{device}"
        return device
    return str(device)


def load_image_for_yolo(path: Path) -> np.ndarray:
    """
    Load image as uint8 HWC for inference. Matches YOLO dataset convention:
    TIFF with CYX (channel-first) from tifffile; otherwise single-page array.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with TiffFile(path) as tif:
            series = tif.series[0]
            image = series.asarray()
            axes = series.axes

        if image.ndim == 2:
            return np.expand_dims(image.astype(np.uint8, copy=False), axis=-1)

        if axes == "CYX":
            image = np.transpose(image, (1, 2, 0))
        elif axes == "YXC":
            pass
        elif image.ndim == 3 and image.shape[0] < min(image.shape[1], image.shape[2]):
            # Heuristic: small leading dim treated as channels (e.g. SYX stored oddly)
            image = np.transpose(image, (1, 2, 0))

        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        return image

    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8, copy=False)


def run_val(args: argparse.Namespace, data_yaml: Path) -> Any:
    from ultralytics import YOLO

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    val_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        split="val",
        plots=args.plots,
        half=args.half,
    )
    if args.save_json:
        val_kwargs["save_json"] = True
    if args.project is not None:
        val_kwargs["project"] = str(args.project.resolve())
        val_kwargs["name"] = args.name
    return model.val(**val_kwargs)


def run_benchmark(args: argparse.Namespace, data_yaml: Path) -> Any:
    from ultralytics.utils.benchmarks import benchmark

    device = _parse_device(args.device)
    return benchmark(
        model=str(Path(args.weights).resolve()),
        data=str(data_yaml),
        imgsz=args.imgsz,
        half=args.half,
        device=device,
        verbose=args.verbose,
        format=args.format.strip(),
    )


def run_sahi(args: argparse.Namespace, data_yaml: Path) -> None:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    dataset_root, config = load_dataset_config_from_yaml(data_yaml)
    samples = collect_samples(dataset_root, config, "val")
    if not samples:
        split_path = config.get("val", "")
        resolved = resolve_split_dir(dataset_root, split_path) if split_path else None
        raise FileNotFoundError(
            f"No val samples with paired labels under {resolved or dataset_root}"
        )

    if args.max_images is not None:
        samples = samples[: args.max_images]

    device = device_for_sahi(_parse_device(args.device))
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(Path(args.weights).resolve()),
        confidence_threshold=args.conf,
        device=device,
    )

    out_dir = args.sahi_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, _label_path in samples:
        image = load_image_for_yolo(image_path)
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            verbose=0,
        )
        subdir = out_dir / image_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        result.export_visuals(export_dir=str(subdir), file_name="prediction_visual")
        print(f"SAHI: wrote {subdir / 'prediction_visual.png'}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_yaml = _resolve_data_yaml(args)

    if args.mode == "val":
        run_val(args, data_yaml)
    elif args.mode == "benchmark":
        run_benchmark(args, data_yaml)
    else:
        run_sahi(args, data_yaml)


if __name__ == "__main__":
    main()
