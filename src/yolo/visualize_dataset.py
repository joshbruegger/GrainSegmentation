import argparse
import random
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image
from matplotlib import patches

from ultralytics.utils.plotting import colors


matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save random annotated YOLO segmentation samples for train and val."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing a YOLO dataset YAML plus images/ and labels/ folders.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where train/ and val/ visualization PNGs will be written.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=4,
        help="Maximum number of random samples to save per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    args = parser.parse_args(argv)
    if args.num < 1:
        parser.error("--num must be at least 1")
    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "visualizations"
    return args


def find_dataset_yaml(dataset_dir: Path) -> Path:
    yaml_paths = sorted(dataset_dir.glob("*.yaml")) + sorted(dataset_dir.glob("*.yml"))
    if not yaml_paths:
        raise FileNotFoundError(f"No dataset YAML found in {dataset_dir}")
    if len(yaml_paths) == 1:
        return yaml_paths[0]

    preferred = dataset_dir / f"{dataset_dir.name}.yaml"
    if preferred.exists():
        return preferred
    raise ValueError(f"Multiple dataset YAML files found in {dataset_dir}")


def load_dataset_config(dataset_dir: Path) -> tuple[Path, dict, dict[int, str]]:
    dataset_yaml = find_dataset_yaml(dataset_dir)
    with dataset_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    names = config.get("names", {})
    if isinstance(names, list):
        label_map = {index: name for index, name in enumerate(names)}
    else:
        label_map = {int(index): name for index, name in names.items()}

    return dataset_root, config, label_map


def resolve_split_dir(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()


def default_label_dir(dataset_root: Path, split_name: str, image_dir: Path) -> Path:
    relative_parts = list(image_dir.relative_to(dataset_root).parts)
    if "images" in relative_parts:
        relative_parts[relative_parts.index("images")] = "labels"
        return dataset_root.joinpath(*relative_parts)
    return dataset_root / "labels" / split_name


def collect_samples(
    dataset_root: Path, config: dict, split_name: str
) -> list[tuple[Path, Path]]:
    split_key = config.get(split_name)
    if not split_key:
        return []

    image_dir = resolve_split_dir(dataset_root, split_key)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory for split '{split_name}': {image_dir}"
        )

    label_dir = default_label_dir(dataset_root, split_name, image_dir)
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append((image_path, label_path))
    return samples


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    channel_min = float(channel.min())
    channel_max = float(channel.max())
    if channel_max <= channel_min:
        return np.zeros_like(channel, dtype=np.uint8)
    scaled = (channel - channel_min) / (channel_max - channel_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_image_preview(image_path: Path) -> tuple[np.ndarray, str | None]:
    with Image.open(image_path) as image:
        frame_count = getattr(image, "n_frames", 1)
        if frame_count > 1:
            channels = []
            for index in range(min(frame_count, 3)):
                image.seek(index)
                channels.append(_normalize_channel(np.asarray(image)))
            while len(channels) < 3:
                channels.append(channels[-1])
            return np.stack(channels[:3], axis=-1), None

        array = np.asarray(image)

    if array.ndim == 2:
        return array, "gray"
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[..., :3], None
    if array.ndim == 3 and array.shape[2] == 1:
        return array[..., 0], "gray"
    raise ValueError(f"Unsupported image shape for visualization: {array.shape}")


def _read_polygons(
    label_path: Path, image_width: int, image_height: int
) -> list[tuple[int, np.ndarray]]:
    polygons: list[tuple[int, np.ndarray]] = []
    with label_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(f"Invalid segmentation label row in {label_path}")
            class_id = int(values[0])
            points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= image_width
            points[:, 1] *= image_height
            polygons.append((class_id, points))
    return polygons


def save_visualization(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    label_map: dict[int, str],
) -> None:
    preview, cmap = load_image_preview(image_path)
    image_height, image_width = preview.shape[:2]
    polygons = _read_polygons(
        label_path, image_width=image_width, image_height=image_height
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview, cmap=cmap)
    ax.axis("off")

    for class_id, points in polygons:
        color = tuple(channel / 255 for channel in colors(class_id, False))
        polygon = patches.Polygon(
            points, closed=True, fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(polygon)

        anchor_x = float(np.min(points[:, 0]))
        anchor_y = float(np.min(points[:, 1]))
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        text_color = "white" if luminance < 0.5 else "black"
        ax.text(
            anchor_x,
            max(anchor_y - 4.0, 0.0),
            label_map.get(class_id, str(class_id)),
            color=text_color,
            backgroundcolor=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_split_visualizations(
    split_name: str,
    samples: list[tuple[Path, Path]],
    output_dir: Path,
    label_map: dict[int, str],
    num_samples: int,
    rng: random.Random,
) -> int:
    if not samples:
        return 0

    selected = rng.sample(samples, k=min(num_samples, len(samples)))
    for index, (image_path, label_path) in enumerate(selected, start=1):
        output_path = output_dir / split_name / f"{index:03d}_{image_path.stem}.png"
        save_visualization(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            label_map=label_map,
        )
    return len(selected)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    dataset_root, config, label_map = load_dataset_config(dataset_dir)
    rng = random.Random(args.seed)

    for split_name in ("train", "val"):
        samples = collect_samples(dataset_root, config, split_name)
        saved_count = save_split_visualizations(
            split_name=split_name,
            samples=samples,
            output_dir=args.output_dir.resolve(),
            label_map=label_map,
            num_samples=args.num,
            rng=rng,
        )
        print(f"Saved {saved_count} visualization(s) for {split_name}")


if __name__ == "__main__":
    main()
