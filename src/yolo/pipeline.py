from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import default_dataset_root, default_run_root, get_variant_config

from ultralytics import settings


@dataclass(frozen=True)
class ResolvedVariantPaths:
    variant_name: str
    dataset_dir: Path
    data_yaml: Path
    channels: int


def default_project_dir(*, scratch_root: str | Path | None = None) -> Path:
    return default_run_root(scratch_root)


def default_resume_checkpoint(*, project_dir: str | Path, run_name: str) -> Path:
    return Path(project_dir) / run_name / "weights" / "last.pt"


def resolve_variant_paths(
    *, variant_name: str, scratch_root: str | Path | None = None
) -> ResolvedVariantPaths:
    variant = get_variant_config(variant_name)
    dataset_dir = default_dataset_root(scratch_root) / variant.dataset_subdir
    return ResolvedVariantPaths(
        variant_name=variant.name,
        dataset_dir=dataset_dir,
        data_yaml=dataset_dir / variant.yaml_name,
        channels=variant.channels,
    )


def _resolve_yolo_factory(yolo_factory):
    if yolo_factory is None:
        from ultralytics import YOLO

        return YOLO
    return yolo_factory


def _resolve_data_yaml(data_yaml: str | Path) -> Path:
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {data_yaml_path}")
    return data_yaml_path


def _serialize_tune_device(device: int | str | list[int]) -> int | str:
    if isinstance(device, list):
        return ",".join(str(part) for part in device)
    return device


def train_model(
    *,
    data_yaml: str | Path,
    run_name: str,
    project_dir: str | Path,
    model_source: str,
    resume_path: str | Path | None,
    epochs: int,
    imgsz: int,
    batch: int | float,
    workers: int,
    device: int | str | list[int],
    cache: bool | str,
    amp: bool,
    plots: bool,
    exist_ok: bool,
    yolo_factory=None,
) -> Any:
    settings.update({"tensorboard": True})
    data_yaml_path = _resolve_data_yaml(data_yaml)
    project_dir_path = Path(project_dir)
    yolo_factory = _resolve_yolo_factory(yolo_factory)

    train_args = dict(
        data=str(data_yaml_path),
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        cache=cache,
        plots=plots,
        optimizer="MuSGD",
        nbs=32,
        warmup_epochs=3,
        patience=50,
        cos_lr=True,
        lr0=0.001,
        dropout=0.05,
        flipud=0.5,
        degrees=180,
    )

    if resume_path is not None:
        resume_checkpoint = Path(resume_path)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        model = yolo_factory(str(resume_checkpoint))
        train_args["resume"] = True
    else:
        model = yolo_factory(model_source)
        train_args["epochs"] = epochs
        train_args["project"] = str(project_dir_path)
        train_args["name"] = run_name
        train_args["amp"] = amp
        train_args["exist_ok"] = exist_ok

    return model.train(**train_args)


def tune_model(
    *,
    data_yaml: str | Path,
    run_name: str,
    project_dir: str | Path,
    model_source: str,
    epochs: int,
    iterations: int,
    imgsz: int,
    batch: int | float,
    workers: int,
    device: int | str | list[int],
    cache: bool | str,
    amp: bool,
    resume: bool,
    exist_ok: bool,
    yolo_factory=None,
) -> Any:
    settings.update({"tensorboard": True})
    data_yaml_path = _resolve_data_yaml(data_yaml)
    project_dir_path = Path(project_dir)
    yolo_factory = _resolve_yolo_factory(yolo_factory)
    model = yolo_factory(model_source)

    search_space = {
        "lr0": (6e-4, 2.5e-3),
        "dropout": (0.1, 0.6),
    }

    tune_args = dict(
        data=str(data_yaml_path),
        epochs=epochs,
        iterations=iterations,
        space=search_space,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=_serialize_tune_device(device),
        cache=cache,
        project=str(project_dir_path),
        name=run_name,
        amp=amp,
        exist_ok=exist_ok,
        optimizer="MuSGD",
        nbs=32,
        warmup_epochs=2,
        patience=10,
        cos_lr=True,
        lr0=1.5e-3,
        dropout=0.35,
        flipud=0.5,
        degrees=180,
    )
    if resume:
        tune_args["resume"] = True

    return model.tune(**tune_args)
