from dataclasses import dataclass
import os
from pathlib import Path


DATASET_ROOT = Path("GrainSeg") / "dataset" / "MWD-1#121" / "yolo"
RUN_ROOT = Path("GrainSeg") / "runs" / "yolo26-seg"


@dataclass(frozen=True)
class VariantConfig:
    name: str
    dataset_subdir: str
    yaml_name: str
    channels: int
    slurm_mem: str
    slurm_gpus_per_node: str = "rtx_pro_6000:2"
    slurm_cpus_per_task: int = 16
    slurm_time: str = "12:00:00"
    default_batch: int = -1
    default_imgsz: int = 1024
    default_workers: int = 16


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "PPL": VariantConfig(
        name="PPL",
        dataset_subdir="PPL",
        yaml_name="PPL.yaml",
        channels=1,
        slurm_mem="256G",
    ),
    "PPLPPXblend": VariantConfig(
        name="PPLPPXblend",
        dataset_subdir="PPLPPXblend",
        yaml_name="PPLPPXblend.yaml",
        channels=1,
        slurm_mem="256G",
    ),
    "PPL+PPXblend": VariantConfig(
        name="PPL+PPXblend",
        dataset_subdir="PPL+PPXblend",
        yaml_name="PPL_PPXblend.yaml",
        channels=6,
        slurm_mem="512G",
    ),
    "PPL+AllPPX": VariantConfig(
        name="PPL+AllPPX",
        dataset_subdir="PPL+AllPPX",
        yaml_name="PPL+AllPPX.yaml",
        channels=32,
        slurm_mem="950G",
    ),
}


def default_scratch_root(scratch_root: str | Path | None = None) -> Path:
    if scratch_root is not None:
        return Path(scratch_root)
    return Path(os.environ.get("SCRATCH", "/scratch"))


def default_dataset_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / DATASET_ROOT


def default_run_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / RUN_ROOT


def get_variant_config(name: str) -> VariantConfig:
    try:
        return VARIANT_CONFIGS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(
            f"Unknown YOLO variant {name!r}. Expected one of: {valid}"
        ) from exc


def variant_choices() -> tuple[str, ...]:
    return tuple(VARIANT_CONFIGS)
