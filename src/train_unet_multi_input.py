import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune multi-input U-Net from raster masks."
    )
    parser.add_argument("--image-dir", required=True, help="Directory of input images")
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing raster masks",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the single-image U-Net checkpoint (.keras)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a 7-input checkpoint to resume training (.keras)",
    )
    parser.add_argument(
        "--output-model", required=True, help="Path to save the fine-tuned model"
    )
    parser.add_argument("--patch-size", type=int, default=3008)
    parser.add_argument("--stride", type=int, default=1504)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img1-suffix", default="_img1")
    parser.add_argument("--img-suffix-template", default="_img{index}")
    parser.add_argument(
        "--mask-ext",
        default=None,
        help="Raster mask file extension (e.g. .png). Defaults to common types.",
    )
    parser.add_argument(
        "--mask-stem-suffix",
        default="",
        help="Optional extra suffix before the raster mask extension",
    )
    parser.add_argument("--num-inputs", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spatial-split",
        action="store_true",
        help="Split each image into grid regions for train/val/test.",
    )
    parser.add_argument(
        "--split-tile-size",
        type=int,
        default=0,
        help="Grid tile size for spatial split (defaults to patch size).",
    )
    parser.add_argument(
        "--split-coverage-bins",
        type=int,
        default=8,
        help="Number of coverage bins for spatial stratification.",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_inputs != 7:
        raise ValueError("This training pipeline expects --num-inputs 7.")
    if args.checkpoint and args.resume:
        raise ValueError("Use only one of --checkpoint or --resume.")
    if args.split_tile_size < 0:
        raise ValueError("--split-tile-size must be >= 0")

    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)
    from segmenteverytrain.train import train_model

    split_tile_size = args.split_tile_size or args.patch_size

    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        output_model_path=args.output_model,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        img1_suffix=args.img1_suffix,
        img_suffix_template=args.img_suffix_template,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        spatial_split=args.spatial_split,
        split_tile_size=split_tile_size,
        split_coverage_bins=args.split_coverage_bins,
        num_inputs=args.num_inputs,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        use_mixed_precision=not args.no_mixed_precision,
    )


if __name__ == "__main__":
    main()
