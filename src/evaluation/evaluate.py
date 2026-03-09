import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Add src to sys.path to import from training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.data import list_samples, _load_rgb_image, _load_raster_mask
from training.model import weighted_crossentropy
from evaluation.inference import predict_full_image
from evaluation.metrics import (
    compute_semantic_metrics,
    compute_boundary_f1,
    compute_boundary_iou,
    get_instances,
    compute_aji,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net model on test images."
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained .keras model"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--mask-dir", required=True, help="Directory containing ground truth masks"
    )
    parser.add_argument(
        "--output-json", required=True, help="Path to save evaluation metrics JSON"
    )
    parser.add_argument(
        "--save-predictions-dir",
        help="Optional directory to save predicted mask images",
    )
    parser.add_argument(
        "--num-inputs", type=int, default=7, help="Number of inputs (1, 2, or 7)"
    )
    parser.add_argument(
        "--image-suffixes",
        nargs="+",
        default=["_PPL", "_PPX1", "_PPX2", "_PPX3", "_PPX4", "_PPX5", "_PPX6"],
    )
    parser.add_argument("--mask-ext", default=None)
    parser.add_argument("--mask-stem-suffix", default="")
    parser.add_argument("--patch-size", type=int, default=3008)
    parser.add_argument("--stride", type=int, default=1504)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    _validate_args(args, parser)
    return args


def _raise_argument_error(message: str, parser: argparse.ArgumentParser | None = None):
    if parser is None:
        raise ValueError(message)
    parser.error(message)


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs not in {1, 2, 7}:
        _raise_argument_error("num_inputs must be one of: 1, 2, 7", parser)
    if len(args.image_suffixes) < args.num_inputs:
        _raise_argument_error(
            "image_suffixes must provide at least num_inputs suffixes", parser
        )
    if args.patch_size <= 0 or args.stride <= 0:
        _raise_argument_error("patch_size and stride must be > 0", parser)
    if args.stride > args.patch_size:
        _raise_argument_error("stride must be <= patch_size", parser)
    if args.batch_size <= 0:
        _raise_argument_error("batch_size must be > 0", parser)
    if not np.isfinite(args.boundary_tolerance) or args.boundary_tolerance < 0:
        _raise_argument_error("boundary_tolerance must be finite and >= 0", parser)


def _validate_sample_data(
    images: list[np.ndarray], mask: np.ndarray, mask_path: str
) -> np.ndarray:
    if not images:
        raise ValueError("Sample must contain at least one input image.")

    expected_shape = images[0].shape
    if len(expected_shape) != 3:
        raise ValueError("All input images must have shape (H, W, C).")
    for img in images[1:]:
        if img.shape != expected_shape:
            raise ValueError("All input images must share the same shape.")

    if mask.ndim != 2:
        raise ValueError(f"Raster mask must be 2D: {mask_path}")

    image_shape = expected_shape[:2]
    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_shape} "
            f"for {mask_path}"
        )

    mask_int = mask.astype(np.int32)
    if not np.all(mask == mask_int) or np.any((mask_int < 0) | (mask_int > 2)):
        raise ValueError(f"Mask values must be in [0, 2] for {mask_path}")

    return mask_int


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(
        args.model_path, custom_objects={"weighted_crossentropy": weighted_crossentropy}
    )

    samples = list_samples(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_suffixes=args.image_suffixes,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        num_inputs=args.num_inputs,
    )

    if not samples:
        print("No samples found. Exiting.")
        sys.exit(1)

    print(f"Found {len(samples)} samples to evaluate.")

    if args.save_predictions_dir:
        os.makedirs(args.save_predictions_dir, exist_ok=True)

    results = {}
    all_metrics = []

    for sample in samples:
        sample_id = sample["id"]
        print(f"Evaluating sample: {sample_id}")

        # Load images
        images = [_load_rgb_image(p) for p in sample["images"]]
        if len(images) != args.num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")
        true_mask = _validate_sample_data(
            images, _load_raster_mask(sample["mask"]), sample["mask"]
        )

        # Predict
        pred_classes, _ = predict_full_image(
            model=model,
            inputs=tuple(images),
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        if args.save_predictions_dir:
            out_img_path = os.path.join(
                args.save_predictions_dir, f"{sample_id}_pred.png"
            )
            Image.fromarray(pred_classes.astype(np.uint8)).save(out_img_path)

        # Compute Metrics
        metrics = compute_semantic_metrics(true_mask, pred_classes, num_classes=3)

        bnd_f1 = compute_boundary_f1(
            true_mask, pred_classes, class_idx=2, tolerance=args.boundary_tolerance
        )
        bnd_iou = compute_boundary_iou(
            true_mask, pred_classes, class_idx=2, tolerance=args.boundary_tolerance
        )

        metrics["boundary_f1"] = bnd_f1
        metrics["boundary_iou"] = bnd_iou

        # Instance metrics
        true_instances = get_instances(true_mask, interior_class=1)
        pred_instances = get_instances(pred_classes, interior_class=1)

        aji = compute_aji(true_instances, pred_instances)
        metrics["aji"] = aji

        results[sample_id] = metrics
        all_metrics.append(metrics)

        print(
            f"Metrics for {sample_id}: IoU_Int: {metrics['iou_class_1']:.4f}, IoU_Bnd: {metrics['iou_class_2']:.4f}, Bnd_F1: {bnd_f1:.4f}, AJI: {aji:.4f}"
        )

    # Compute means
    mean_metrics = {}
    for k in all_metrics[0].keys():
        values = [m[k] for m in all_metrics if not np.isnan(m[k])]
        mean_metrics[k] = float(np.mean(values)) if values else float("nan")

    results["mean"] = mean_metrics

    print("\n--- Mean Metrics ---")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
