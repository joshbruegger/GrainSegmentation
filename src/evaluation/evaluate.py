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
from evaluation.coco_mask_ap import (
    aggregate_coco_means,
    evaluate_mask_ap,
    instance_label_map_to_coco_dt,
    instance_label_map_to_coco_gt,
)
from evaluation.inference import predict_full_image
from evaluation.instance_masks import (
    semantic_to_instance_label_map,
    semantic_to_instance_label_map_watershed,
)
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
    parser.add_argument(
        "--coco-mask-ap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also compute COCO mask AP from instance label maps (GT and pred).",
    )
    parser.add_argument(
        "--instance-method",
        choices=("cc", "watershed"),
        default="cc",
        help="How to derive predicted instances for AJI and COCO AP.",
    )
    parser.add_argument(
        "--watershed-min-distance",
        type=int,
        default=1,
        help="peak_local_max min_distance when --instance-method watershed.",
    )
    parser.add_argument(
        "--watershed-boundary-dilate-iter",
        type=int,
        default=0,
        help="Binary dilation iterations on boundary mask for watershed ridge.",
    )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        choices=(1, 2),
        default=1,
        help="skimage watershed connectivity (1 or 2) when --instance-method watershed.",
    )
    parser.add_argument(
        "--watershed-min-area-px",
        type=int,
        default=0,
        help="Drop instances smaller than this many pixels (0 disables).",
    )
    parser.add_argument(
        "--watershed-exclude-border",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="peak_local_max exclude_border when --instance-method watershed.",
    )
    parser.add_argument(
        "--watershed-ridge-level",
        type=float,
        default=None,
        help=(
            "Ridge elevation for boundary; omit for automatic (matches tuning JSON ridge_level null)."
        ),
    )
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
    if args.watershed_min_distance < 1:
        _raise_argument_error("watershed_min_distance must be >= 1", parser)
    if args.watershed_boundary_dilate_iter < 0:
        _raise_argument_error("watershed_boundary_dilate_iter must be >= 0", parser)
    if args.watershed_min_area_px < 0:
        _raise_argument_error("watershed_min_area_px must be >= 0", parser)
    if args.watershed_ridge_level is not None and not np.isfinite(
        args.watershed_ridge_level
    ):
        _raise_argument_error("watershed_ridge_level must be finite when set", parser)


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


# Averaged only via aggregate_coco_means when --coco-mask-ap (skips -1 sentinels).
_COCO_SCALAR_KEYS = frozenset(
    ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100")
)


def _compute_mean_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    mean_metrics = {}
    for key in all_metrics[0].keys():
        if key in _COCO_SCALAR_KEYS:
            continue
        values: list[float] = []
        for metrics in all_metrics:
            v = metrics.get(key)
            if v is None or isinstance(v, (list, dict)):
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                fv = float(v)
                if not np.isnan(fv):
                    values.append(fv)
        mean_metrics[key] = float(np.mean(values)) if values else float("nan")
    return mean_metrics


def _build_results_payload(
    sample_results: dict[str, dict[str, float]], all_metrics: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    results = dict(sample_results)
    if len(all_metrics) > 1:
        results["mean"] = _compute_mean_metrics(all_metrics)
    return results


def _pred_instances_for_metrics(
    pred_classes: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """Instance label map for predicted-instance metrics; GT always uses connected components."""
    if args.instance_method == "cc":
        return semantic_to_instance_label_map(pred_classes, min_area_px=0)
    return semantic_to_instance_label_map_watershed(
        pred_classes,
        min_distance=args.watershed_min_distance,
        boundary_dilate_iter=args.watershed_boundary_dilate_iter,
        watershed_connectivity=args.watershed_connectivity,
        min_area_px=args.watershed_min_area_px,
        exclude_border=args.watershed_exclude_border,
        ridge_level=args.watershed_ridge_level,
    )


def _print_summary(results: dict[str, dict[str, float]], sample_count: int) -> None:
    if sample_count == 1:
        print("\n--- Single-Sample Evaluation ---")
        print(
            "Descriptive only: one evaluation sample found; skipping aggregate mean metrics."
        )
        return

    mean_metrics = results["mean"]
    print("\n--- Mean Metrics ---")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")


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
    coco_per_image: list[dict[str, float]] = []

    for image_idx, sample in enumerate(samples, start=1):
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
        pred_instances = _pred_instances_for_metrics(pred_classes, args)

        aji = compute_aji(true_instances, pred_instances)
        metrics["aji"] = aji

        if args.coco_mask_ap:
            h, w = int(true_mask.shape[0]), int(true_mask.shape[1])
            gt_for_coco = true_instances
            pred_for_coco = pred_instances
            gt_anns = instance_label_map_to_coco_gt(
                gt_for_coco, image_id=image_idx, height=h, width=w
            )
            dt_anns = instance_label_map_to_coco_dt(
                pred_for_coco, image_id=image_idx, height=h, width=w
            )
            coco_summary = evaluate_mask_ap(
                image_id=image_idx,
                file_name=f"{sample_id}_eval.png",
                height=h,
                width=w,
                gt_annotations=gt_anns,
                dt_annotations=dt_anns,
            )
            coco_dict = coco_summary.to_dict()
            metrics.update(coco_dict)
            coco_per_image.append(coco_dict)

        results[sample_id] = metrics
        all_metrics.append(metrics)

        line = (
            f"Metrics for {sample_id}: IoU_Int: {metrics['iou_class_1']:.4f}, "
            f"IoU_Bnd: {metrics['iou_class_2']:.4f}, Bnd_F1: {bnd_f1:.4f}, AJI: {aji:.4f}"
        )
        if args.coco_mask_ap:
            line += (
                f", AP: {metrics['AP']:.4f}, AP50: {metrics['AP50']:.4f}, "
                f"AP75: {metrics['AP75']:.4f}"
            )
        print(line)

    results = _build_results_payload(results, all_metrics)
    if args.coco_mask_ap and len(coco_per_image) > 1:
        if "mean" not in results:
            results["mean"] = {}
        results["mean"].update(aggregate_coco_means(coco_per_image))
    _print_summary(results, len(all_metrics))

    # Save to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
