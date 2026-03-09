import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
        value is not None for value in (args.json_files, args.labels, args.output_plot)
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


def generate_quantitative_plot(json_files, labels, output_path):
    metrics_to_plot = {
        "Mask IoU (Interior)": "iou_class_1",
        "Mask IoU (Boundary)": "iou_class_2",
        "Boundary F1": "boundary_f1",
        "AJI": "aji",
    }

    means = {m: [] for m in metrics_to_plot}
    cis = {m: [] for m in metrics_to_plot}

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        # extract per-sample metrics to compute confidence intervals
        sample_keys = [k for k in data.keys() if k != "mean"]

        for m_name, m_key in metrics_to_plot.items():
            vals = [
                data[sk][m_key] for sk in sample_keys if not np.isnan(data[sk][m_key])
            ]
            means[m_name].append(np.mean(vals))
            cis[m_name].append(compute_ci(vals))

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width

        m_means = [means[m][i] for m in metrics_to_plot]
        m_cis = [cis[m][i] for m in metrics_to_plot]

        ax.bar(x + offset, m_means, width, yerr=m_cis, label=label, capsize=5)

    ax.set_ylabel("Score")
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
    # Create an RGB color mask: background=0,0,0, interior=blue, boundary=red
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = [0.0, 0.0, 1.0]  # Interior
    color_mask[mask == 2] = [1.0, 0.0, 0.0]  # Boundary

    # Where mask > 0, blend image and color
    alpha = 0.4
    overlay = np.copy(image)
    active = mask > 0
    overlay[active] = image[active] * (1 - alpha) + color_mask[active] * alpha
    return overlay


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

    num_cols = 2 + len(preds)

    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original PPL")
    axes[0].axis("off")

    axes[1].imshow(blend_overlay(rgb_img, gt_mask))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    for i, (pred, label) in enumerate(zip(preds, labels)):
        axes[2 + i].imshow(blend_overlay(rgb_img, pred))
        axes[2 + i].set_title(f"Pred: {label}")
        axes[2 + i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved qualitative overlay to {output_path}")


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
