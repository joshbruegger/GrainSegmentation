from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import default_dataset_root, default_run_root, get_variant_config


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
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {data_yaml_path}")

    project_dir_path = Path(project_dir)
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    if resume_path is not None:
        resume_checkpoint = Path(resume_path)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        model = yolo_factory(str(resume_checkpoint))
        return model.train(
            resume=True,
            data=str(data_yaml_path),
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            cache=cache,
            plots=plots,
        )

    model = yolo_factory(model_source)
    return model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        cache=cache,
        project=str(project_dir_path),
        name=run_name,
        amp=amp,
        plots=plots,
        exist_ok=exist_ok,
    )
