import argparse
import os
import sys
from train import train_model


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
        help="Path to a checkpoint to initialize weights (.keras).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume training (.keras).",
    )
    parser.add_argument(
        "--output-model", required=True, help="Path to save the fine-tuned model"
    )
    parser.add_argument("--patch-size", type=int, default=3008)
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=0.5,
        help="Percentage to overlap patches (0 to 1)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--tune-epochs", type=int, default=20, help="Max epochs per tuning trial"
    )
    parser.add_argument(
        "--run-name",
        default="default_run",
        help="Name of the run, used to separate tuning directories.",
    )
    parser.add_argument(
        "--tuning-dir",
        default="tuning_dir",
        help="Base directory for hyperparameter tuning logs.",
    )
    parser.add_argument(
        "--image-suffixes",
        nargs="+",
        default=["_PPL", "_PPX1", "_PPX2", "_PPX3", "_PPX4", "_PPX5", "_PPX6"],
        help="List of space-separated image suffixes to load",
    )
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
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=7,
        help="Number of input images (1=PPL, 2=PPL+composite, 7=PPL+all PPX).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=2,
        help="Number of spatial folds for cross-validation",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=7,
        help="Maximum number of hyperparameter tuning trials",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning and use sensible defaults",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_inputs not in {1, 2, 7}:
        raise ValueError("--num-inputs must be one of: 1, 2, 7.")
    if args.checkpoint and args.resume:
        raise ValueError("Use only one of --checkpoint or --resume.")
    if args.split_tile_size < 0:
        raise ValueError("--split-tile-size must be >= 0")
    if args.patch_overlap < 0 or args.patch_overlap >= 1:
        raise ValueError("--patch-overlap must be in [0, 1).")

    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)

    split_tile_size = args.split_tile_size or args.patch_size
    stride = int(args.patch_size * (1 - args.patch_overlap))

    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        output_model_path=args.output_model,
        patch_size=args.patch_size,
        stride=stride,
        tune_epochs=args.tune_epochs,
        final_epochs=args.epochs,
        image_suffixes=[s.strip() for s in args.image_suffixes],
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        split_tile_size=split_tile_size,
        split_coverage_bins=args.split_coverage_bins,
        num_inputs=args.num_inputs,
        run_name=args.run_name,
        tuning_dir=args.tuning_dir,
        n_splits=args.folds,
        random_state=args.seed,
        use_mixed_precision=not args.no_mixed_precision,
        max_trials=args.max_trials,
        skip_tuning=args.skip_tuning,
    )


if __name__ == "__main__":
    main()
