import argparse
from pathlib import Path
import sys

from config import variant_choices
from pipeline import (
    default_project_dir,
    default_resume_checkpoint,
    resolve_variant_paths,
    train_model,
)


def _parse_device(value: str) -> int | str | list[int]:
    normalized = value.strip()
    if "," in normalized:
        return [int(part.strip()) for part in normalized.split(",") if part.strip()]
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    return normalized


def _print_start_message(args: argparse.Namespace, *, data_yaml: Path) -> None:
    fields = [
        ("variant", args.variant),
        ("data", data_yaml),
        ("name", args.name),
        ("project", args.project),
        ("weights", args.weights),
        ("resume", args.resume),
        ("resume_checkpoint", args.resume_checkpoint),
        ("epochs", args.epochs),
        ("imgsz", args.imgsz),
        ("batch", args.batch),
        ("workers", args.workers),
        ("device", args.device),
        ("cache", args.cache),
        ("amp", args.amp),
        ("plots", args.plots),
    ]
    width = max(len(key) for key, _ in fields)
    border = "=" * 80
    print(border)
    print("YOLO26 Seg Training Start")
    print(border)
    for key, value in fields:
        print(f"{key:<{width}} : {value}")
    print(border)


def _argv_contains(argv: list[str], *options: str) -> bool:
    return any(option in argv for option in options)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume a GrainSegmentation yolo26x-seg run."
    )
    parser.add_argument(
        "--variant",
        choices=variant_choices(),
        help="Named dataset variant to train (for example PPL or PPL+AllPPX).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional explicit path to a YOLO dataset YAML override.",
    )
    parser.add_argument(
        "--name",
        "--run-name",
        dest="name",
        default=None,
        help="Training run name under the project directory.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Output project directory for Ultralytics training artifacts.",
    )
    parser.add_argument(
        "--weights",
        default="yolo26x-seg.pt",
        help="Model weights or YAML to load for a fresh run.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--device",
        default="0,1",
        help="Ultralytics device value, for example 0, 0,1, cpu, or -1.",
    )
    parser.add_argument("--cache", default="disk")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the default last.pt under project/name. Per the indexed "
            "@Yolo docs, saved Ultralytics training state is restored and some "
            "fresh-run overrides may be ignored."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Explicit checkpoint path to resume from. Per the indexed @Yolo docs, "
            "resume restores saved training state and some fresh-run overrides may "
            "be ignored."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)
    if not args.variant and not args.data:
        parser.error("one of --variant or --data is required")
    return args


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    if args.resume and args.resume_checkpoint:
        raise ValueError("Use only one of --resume or --resume-checkpoint.")
    if args.resume or args.resume_checkpoint:
        if _argv_contains(raw_argv, "--epochs"):
            raise ValueError(
                "Ultralytics does not support overriding --epochs while resuming."
            )
        if _argv_contains(raw_argv, "--amp", "--no-amp"):
            raise ValueError(
                "Ultralytics does not support overriding --amp while resuming."
            )

    run_name = args.name or args.variant or Path(args.data).stem
    project_dir = Path(args.project) if args.project else default_project_dir()

    if args.data:
        data_yaml = Path(args.data)
    else:
        data_yaml = resolve_variant_paths(variant_name=args.variant).data_yaml

    resume_path = None
    if args.resume:
        resume_path = default_resume_checkpoint(
            project_dir=project_dir, run_name=run_name
        )
    elif args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)

    args.name = run_name
    args.project = project_dir
    args.device = _parse_device(args.device)
    _print_start_message(args, data_yaml=data_yaml)

    train_model(
        data_yaml=data_yaml,
        run_name=run_name,
        project_dir=project_dir,
        model_source=args.weights,
        resume_path=resume_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=int(args.batch) if args.batch == int(args.batch) else args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        amp=args.amp,
        plots=args.plots,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
