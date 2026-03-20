"""
Evaluate trained YOLO segmentation models: Ultralytics val or SAHI tiled inference on held-out TIFFs (COCO mask AP).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tifffile import TiffFile

from config import variant_choices
from pipeline import resolve_variant_paths
from train import _parse_device

from coco_instance_ap import (
    build_gt_annotations,
    evaluate_mask_ap,
    load_polygons_from_gpkg,
    normalize_polygons_to_image_space,
    object_predictions_to_coco_dt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 segmentation evaluation: val or sahi (whole held-out TIFF + COCO mask AP)."
    )
    parser.add_argument(
        "--mode",
        choices=("val", "sahi"),
        required=True,
        help=(
            "val: Ultralytics validator on dataset test split; sahi: whole held-out TIFF + COCO mask AP vs GPKG."
        ),
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
        default="test",
        help="Run name under project for val-mode (test split) outputs.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FP16 for val (when supported).",
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
    # sahi (tiled inference on whole TIFFs)
    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1024)
    parser.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.2,
        help="Slice overlap ratio (sahi mode).",
    )
    parser.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.2,
        help="Slice overlap ratio (sahi mode).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for SAHI AutoDetectionModel.",
    )
    parser.add_argument(
        "--test-tiff",
        type=Path,
        default=None,
        help="Held-out GeoTIFF path (required for sahi unless --manifest).",
    )
    parser.add_argument(
        "--test-gpkg",
        type=Path,
        default=None,
        help="Ground-truth GeoPackage with grain polygons (sahi mode).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON list of {test_tiff, test_gpkg} pairs for batch sahi evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write sahi metrics JSON to this path.",
    )
    parser.add_argument(
        "--sahi-out-dir",
        type=Path,
        default=None,
        help="Optional: save SAHI prediction_visual.png per dataset under this directory.",
    )

    args = parser.parse_args(argv)
    if args.mode == "sahi":
        if args.manifest is not None:
            pass
        elif args.test_tiff is not None and args.test_gpkg is not None:
            pass
        else:
            parser.error("sahi requires --manifest or both --test-tiff and --test-gpkg")
    elif not args.variant and not args.data:
        parser.error("one of --variant or --data is required")
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
        split="test",
        plots=args.plots,
        half=args.half,
    )
    if args.save_json:
        val_kwargs["save_json"] = True
    if args.project is not None:
        val_kwargs["project"] = str(args.project.resolve())
        val_kwargs["name"] = args.name
    return model.val(**val_kwargs)


def _load_sahi_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.manifest is not None:
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        pairs: list[tuple[Path, Path]] = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"manifest[{index}] must be an object")
            tiff = entry.get("test_tiff") or entry.get("tiff")
            gpkg = entry.get("test_gpkg") or entry.get("gpkg")
            if not tiff or not gpkg:
                raise ValueError(
                    f"manifest[{index}] needs test_tiff and test_gpkg keys"
                )
            pairs.append((Path(tiff).resolve(), Path(gpkg).resolve()))
        return pairs
    return [(args.test_tiff.resolve(), args.test_gpkg.resolve())]


def run_sahi(args: argparse.Namespace) -> dict[str, Any]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    pairs = _load_sahi_pairs(args)
    device = device_for_sahi(_parse_device(args.device))
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(Path(args.weights).resolve()),
        confidence_threshold=args.conf,
        device=device,
    )

    per_image: list[dict[str, Any]] = []
    for image_id, (tiff_path, gpkg_path) in enumerate(pairs, start=1):
        if not tiff_path.is_file():
            raise FileNotFoundError(f"test TIFF not found: {tiff_path}")
        if not gpkg_path.is_file():
            raise FileNotFoundError(f"test GPKG not found: {gpkg_path}")

        image = load_image_for_yolo(tiff_path)
        height, width = image.shape[:2]
        polygons = normalize_polygons_to_image_space(load_polygons_from_gpkg(gpkg_path))
        gt_anns = build_gt_annotations(
            polygons,
            image_id=image_id,
            height=height,
            width=width,
        )
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            verbose=0,
        )
        dt_anns = object_predictions_to_coco_dt(
            result.object_prediction_list,
            image_id=image_id,
            height=height,
            width=width,
        )
        if args.sahi_out_dir is not None:
            out_root = args.sahi_out_dir.resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            sub = out_root / tiff_path.stem
            sub.mkdir(parents=True, exist_ok=True)
            result.export_visuals(export_dir=str(sub), file_name="prediction_visual")

        summary = evaluate_mask_ap(
            image_id=image_id,
            file_name=tiff_path.name,
            height=height,
            width=width,
            gt_annotations=gt_anns,
            dt_annotations=dt_anns,
        )
        row: dict[str, Any] = {
            "test_tiff": str(tiff_path),
            "test_gpkg": str(gpkg_path),
            "image_id": image_id,
            "gt_instances": len(gt_anns),
            "pred_instances": len(dt_anns),
        }
        row.update(summary.to_dict())
        per_image.append(row)
        print(
            f"{tiff_path.name}: AP={summary.ap_50_95:.4f} AP50={summary.ap_50:.4f} "
            f"GT={len(gt_anns)} Pred={len(dt_anns)}"
        )

    mean_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
    )
    aggregate: dict[str, Any] = {"per_image": per_image}
    if len(per_image) == 1:
        for key in mean_keys:
            aggregate[f"mean_{key}"] = float(per_image[0][key])
    else:
        for key in mean_keys:
            values = [
                float(row[key])
                for row in per_image
                if np.isfinite(row[key]) and row[key] >= 0
            ]
            aggregate[f"mean_{key}"] = (
                float(np.mean(values)) if values else float("nan")
            )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON to {args.output_json}")
    return aggregate


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode == "sahi":
        run_sahi(args)
        return

    data_yaml = _resolve_data_yaml(args)
    run_val(args, data_yaml)


if __name__ == "__main__":
    main()
