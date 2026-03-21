import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OVERLAY_MAX_DIM = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quantitative plots and qualitative overlays."
    )
    parser.add_argument(
        "--json-files", nargs="+", help="List of JSON files from evaluate.py"
    )
    parser.add_argument(
        "--labels", nargs="+", help="Labels for the models corresponding to json-files"
    )
    parser.add_argument("--output-plot", help="Path to save the bar chart plot")

    parser.add_argument(
        "--image-path", help="Original PPL input image path (for overlay)"
    )
    parser.add_argument("--gt-path", help="Ground truth mask path")
    parser.add_argument(
        "--pred-paths",
        nargs="+",
        help="Paths to predicted masks corresponding to labels",
    )
    parser.add_argument(
        "--output-overlay", help="Path to save the overlay comparison image"
    )
    args = parser.parse_args()
    _validate_args(args, parser)
    return args


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    quantitative_selected = any(
        value is not None for value in (args.json_files, args.output_plot)
    )
    overlay_selected = any(
        value is not None
        for value in (
            args.image_path,
            args.gt_path,
            args.pred_paths,
            args.output_overlay,
        )
    )

    if not quantitative_selected and not overlay_selected:
        parser.error("Provide either quantitative plot arguments or overlay arguments.")

    if quantitative_selected and not (
        args.json_files and args.labels and args.output_plot
    ):
        parser.error(
            "Quantitative mode requires --json-files, --labels, and --output-plot."
        )

    if overlay_selected and not (
        args.image_path
        and args.gt_path
        and args.pred_paths
        and args.labels
        and args.output_overlay
    ):
        parser.error(
            "Overlay mode requires --image-path, --gt-path, --pred-paths, --labels, and --output-overlay."
        )

    if quantitative_selected and len(args.json_files) != len(args.labels):
        parser.error("Number of json files must match number of labels.")

    if overlay_selected and len(args.pred_paths) != len(args.labels):
        parser.error("Number of pred paths must match number of labels.")


def compute_ci(data, confidence=0.95):
    import scipy.stats as st

    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return 0.0
    se = st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def _load_quantitative_metrics(json_files, metrics_to_plot):
    means = {metric_name: [] for metric_name in metrics_to_plot}
    cis = {metric_name: [] for metric_name in metrics_to_plot}
    sample_counts = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        sample_keys = [k for k in data.keys() if k != "mean"]
        sample_counts.append(len(sample_keys))

        for metric_name, metric_key in metrics_to_plot.items():
            vals = [
                data[sk][metric_key]
                for sk in sample_keys
                if not np.isnan(data[sk][metric_key])
            ]
            means[metric_name].append(np.mean(vals))
            cis[metric_name].append(compute_ci(vals))

    return means, cis, sample_counts


def generate_quantitative_plot(json_files, labels, output_path):
    metrics_to_plot = {
        "AJI": "aji",
        "F1 @ IoU 0.5": "f1_iou50",
        "F1 @ IoU 0.75": "f1_iou75",
        "mF1 @ IoU 0.5:0.95": "mF1_iou50_95",
    }

    means, cis, sample_counts = _load_quantitative_metrics(json_files, metrics_to_plot)
    single_sample_mode = all(count == 1 for count in sample_counts)

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width

        m_means = [means[m][i] for m in metrics_to_plot]
        if single_sample_mode:
            ax.bar(x + offset, m_means, width, label=label)
        else:
            m_cis = [cis[m][i] for m in metrics_to_plot]
            ax.bar(x + offset, m_means, width, yerr=m_cis, label=label, capsize=5)

    ax.set_ylabel("Score")
    if single_sample_mode:
        ax.set_title(
            "Quantitative Ablation Results (descriptive single-image comparison)"
        )
        print(
            "Single-sample input detected; plotting descriptive scores without confidence intervals."
        )
    else:
        ax.set_title("Quantitative Ablation Results")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_to_plot.keys()))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved quantitative plot to {output_path}")


def blend_overlay(image, mask):
    # image: RGB [0, 1], mask: integer 0, 1, 2
    # Use a single red tint for all foreground classes so overlays remain visible.
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [1.0, 0.0, 0.0]

    # Where mask > 0, blend image and color
    alpha = 0.4
    overlay = np.copy(image)
    active = mask > 0
    overlay[active] = image[active] * (1 - alpha) + color_mask[active] * alpha
    return overlay


def _resize_overlay_arrays(rgb_img, gt_mask, preds, max_dim=OVERLAY_MAX_DIM):
    height, width = rgb_img.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return rgb_img, gt_mask, preds

    scale = max_dim / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_size = (resized_width, resized_height)

    resized_image = (
        np.asarray(
            Image.fromarray((rgb_img * 255.0).astype(np.uint8), mode="RGB").resize(
                resized_size, resample=Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    resized_gt = np.asarray(
        Image.fromarray(gt_mask).resize(resized_size, resample=Image.Resampling.NEAREST)
    )
    resized_preds = [
        np.asarray(
            Image.fromarray(pred).resize(
                resized_size, resample=Image.Resampling.NEAREST
            )
        )
        for pred in preds
    ]
    return resized_image, resized_gt, resized_preds


def _sanitize_overlay_label(label: str) -> str:
    safe_chars = []
    for char in label:
        if char.isalnum() or char in {"+", "-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "model"


def _build_overlay_output_paths(
    output_path: str, labels: list[str]
) -> tuple[str, list[str]]:
    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.splitext(os.path.basename(output_path))[0] or "overlay"

    gt_output = os.path.join(output_dir, f"{base_name}_ground_truth.png")
    pred_outputs = [
        os.path.join(output_dir, f"{base_name}_{_sanitize_overlay_label(label)}.png")
        for label in labels
    ]
    return gt_output, pred_outputs


def generate_qualitative_overlay(image_path, gt_path, pred_paths, labels, output_path):
    with Image.open(image_path) as img:
        rgb_img = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    with Image.open(gt_path) as img:
        if img.mode not in ("L", "I", "I;16", "F"):
            img = img.convert("L")
        gt_mask = np.asarray(img)

    preds = []
    for pp in pred_paths:
        with Image.open(pp) as img:
            if img.mode not in ("L", "I", "I;16", "F"):
                img = img.convert("L")
            preds.append(np.asarray(img))

    rgb_img, gt_mask, preds = _resize_overlay_arrays(rgb_img, gt_mask, preds)
    gt_output_path, pred_output_paths = _build_overlay_output_paths(output_path, labels)

    if os.path.exists(output_path):
        os.remove(output_path)

    Image.fromarray((blend_overlay(rgb_img, gt_mask) * 255.0).astype(np.uint8)).save(
        gt_output_path
    )
    print(f"Saved qualitative overlay to {gt_output_path}")

    for label, pred, pred_output_path in zip(labels, preds, pred_output_paths):
        Image.fromarray((blend_overlay(rgb_img, pred) * 255.0).astype(np.uint8)).save(
            pred_output_path
        )
        print(f"Saved qualitative overlay to {pred_output_path} ({label})")


def main():
    args = parse_args()

    if args.json_files and args.labels and args.output_plot:
        generate_quantitative_plot(args.json_files, args.labels, args.output_plot)

    if (
        args.image_path
        and args.gt_path
        and args.pred_paths
        and args.labels
        and args.output_overlay
    ):
        generate_qualitative_overlay(
            args.image_path,
            args.gt_path,
            args.pred_paths,
            args.labels,
            args.output_overlay,
        )


if __name__ == "__main__":
    main()
