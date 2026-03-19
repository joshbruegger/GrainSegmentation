#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from dataclasses import dataclass

import matplotlib  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib import colors  # type: ignore[import-not-found]
from matplotlib.ticker import ScalarFormatter  # type: ignore[import-not-found]

matplotlib.use("Agg")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine tune_results.csv files from multiple runs and plot "
            "smoothed fitness over iteration."
        )
    )
    parser.add_argument(
        "--runs-root",
        default="/scratch/s4361687/GrainSeg/runs/yolo26-seg-tuning",
        help="Root directory containing run folders with tune_results.csv files.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*/ */tune_results.csv".replace(" ", ""),
        help=(
            "Glob pattern (relative to --runs-root) for locating tune_results.csv files. "
            "Default matches layouts like RUN/RUN/tune_results.csv."
        ),
    )
    parser.add_argument(
        "--fitness-col",
        default="fitness",
        help="Column name containing fitness values.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Centered Gaussian smoothing kernel length (odd positive integer).",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also draw unsmoothed series as faint lines.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. Defaults to '<runs-root>/tune_fitness_over_iteration_all_runs.png'."
        ),
    )
    return parser


@dataclass(frozen=True)
class TunePoint:
    lr0: float
    dropout: float
    fitness: float


def _read_fitness_series(csv_path: Path, fitness_col: str) -> np.ndarray:
    values: list[float] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if fitness_col not in reader.fieldnames:
            raise ValueError(
                f"Missing '{fitness_col}' in {csv_path}. Columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            try:
                value = float(row[fitness_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid '{fitness_col}' at {csv_path}:{row_idx}"
                ) from exc
            if not np.isnan(value):
                values.append(value)

    if not values:
        raise ValueError(f"No valid fitness values found in {csv_path}")

    return np.asarray(values, dtype=float)


def _gaussian_smooth_centered(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth values with a centered Gaussian kernel.

    We interpret `window` as the kernel length (must be odd for a centered kernel).
    """
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd for centered smoothing.")
    if window > values.size:
        return np.full_like(values, float(np.mean(values)))

    # Use a wider Gaussian so the result is visibly smoother than a moving
    # average, then apply the filter twice for stronger smoothing.
    sigma = float(window) / 7.0

    pad = window // 2
    x = np.arange(window, dtype=float) - float(pad)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= float(kernel.sum())

    passes = 2 if window >= 5 else 1
    smoothed = values
    for _ in range(passes):
        padded = np.pad(smoothed, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def _discover_csvs(runs_root: Path, glob_pattern: str) -> list[Path]:
    csvs = sorted(runs_root.glob(glob_pattern))
    return [p for p in csvs if p.is_file()]


def _plot_fitness_vs_iterations(csv_paths, runs_root, args, output_path):

    fig, ax = plt.subplots(figsize=(12, 7))

    xmin, xmax = 0, 0
    ymin, ymax = 0, 0

    for csv_path in csv_paths:
        run_name = _infer_run_name(csv_path, runs_root)
        raw = _read_fitness_series(csv_path, args.fitness_col)
        smoothed = _gaussian_smooth_centered(raw, args.smooth_window)
        iterations = np.arange(1, raw.size + 1, dtype=int)

        if args.show_raw:
            ax.scatter(iterations, raw, s=5, alpha=0.25)

        ax.plot(iterations, smoothed, linewidth=2.2, label=run_name)

        # Mark the iteration of the best smoothed fitness for each run.
        best_idx = int(np.argmax(smoothed))
        best_idx_raw = int(np.argmax(raw))
        best_iter_raw = int(iterations[best_idx_raw])
        best_iter = int(iterations[best_idx])
        best_fit_smoothed = float(smoothed[best_idx])
        # Label the raw fitness, not the smoothed value.
        best_fit_raw = float(raw[best_idx_raw])
        ax.scatter(
            best_iter,
            best_fit_smoothed,
            marker="+",
            s=220,
            linewidths=1.5,
            c="black",
            alpha=0.8,
            zorder=10,
        )
        # Label the best point next to the "+" marker.
        # For specific runs, place the label below the "+" to avoid overlap.
        place_below = run_name in {"PPL", "PPL+XPL-Comp"}
        xytext = (4, -15) if place_below else (0, 12)

        va = "top" if place_below else "bottom"
        ha = "center"
        ax.annotate(
            f"iter={best_iter_raw}\nfit={best_fit_raw:.4g}",
            xy=(best_iter, best_fit_smoothed),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color="black",
            zorder=11,
        )

        xmin = min(xmin, float(min(iterations)))
        xmax = max(xmax, float(max(iterations)))
        ymin = min(ymin, float(min(smoothed)))
        ymax = max(ymax, float(max(smoothed)))

    ax.set_xlim(xmin, xmax + 6)
    # ax.set_ylim(ymin, ymax)

    # ax.set_title("Smoothed Fitness over Tuning Iteration (All Variants)")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Smoothed Fitness")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved combined plot to: {output_path}")


def _read_tune_csv(csv_path: str) -> list[TunePoint]:
    points: list[TunePoint] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears to have no header: {csv_path}")

        missing = [
            c for c in ("fitness", "lr0", "dropout") if c not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing}. Found columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):  # header is line 1
            try:
                lr0 = float(row["lr0"])
                dropout = float(row["dropout"])
                fitness = float(row["fitness"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse floats in {csv_path} at row {row_idx}: {e}"
                ) from e

            if not (
                os.path.exists(csv_path) and fitness == fitness
            ):  # fitness is NaN-safe
                # Keep behavior simple; NaN filtered below.
                pass

            if any(v != v for v in (lr0, dropout, fitness)):  # NaN check
                continue

            points.append(TunePoint(lr0=lr0, dropout=dropout, fitness=fitness))

    return points


def _global_scatter_fitness_limits(
    csv_paths: list[Path], percentile_lo: float = 5, percentile_hi: float = 95
) -> tuple[float, float]:
    """Fitness vmin/vmax for scatter colormap, shared across all runs."""
    all_z: list[float] = []
    for p in csv_paths:
        pts = _read_tune_csv(str(p))
        all_z.extend(pp.fitness for pp in pts)
    if not all_z:
        raise ValueError("No fitness values found for global scatter color limits.")
    arr = np.asarray(all_z, dtype=float)
    z_lo = float(np.percentile(arr, percentile_lo))
    z_hi = float(np.percentile(arr, percentile_hi))
    if z_lo == z_hi:
        z_lo, z_hi = float(arr.min()), float(arr.max())
    return z_lo, z_hi


def _plot_tune_scatter(
    csv_path: str,
    run_name: str,
    output_path: str,
    *,
    z_lo: float,
    z_hi: float,
):
    points = _read_tune_csv(csv_path)
    if not points:
        raise ValueError("No valid rows found in CSV after parsing/filtering.")

    xs = [p.lr0 for p in points]
    ys = [p.dropout for p in points]
    zs = [p.fitness for p in points]

    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))

    # x: dropout, y: lr0, color intensity: fitness.
    fig, ax = plt.subplots(figsize=(10, 7))

    norm = colors.Normalize(vmin=z_lo, vmax=z_hi)

    sc = ax.scatter(
        ys,
        xs,
        c=zs,
        # cmap=args.cmap,
        norm=norm,
        alpha=min(1.0, 0.8),
        s=70,
        linewidths=0,
    )
    # Pad axes so markers sitting at the exact data min/max are not clipped.
    dropout_span = y_max - y_min
    lr0_span = x_max - x_min
    dropout_pad = max(1e-12, dropout_span) * 0.02
    lr0_pad = max(1e-12, lr0_span) * 0.02
    ax.set_xlim(y_min - dropout_pad, y_max + dropout_pad)
    ax.set_ylim(x_min - lr0_pad, x_max + lr0_pad)
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Initial Learning Rate")
    # Use scientific notation for lr0 tick labels (e.g., 1.5e-3).
    # Render the exponent using mathtext so ticks look like "x10-3"
    # (Matplotlib's "×10^{-3}" representation).
    lr0_formatter = ScalarFormatter(useMathText=True)
    lr0_formatter.set_scientific(True)
    lr0_formatter.set_powerlimits((0, 0))  # force scientific notation
    ax.yaxis.set_major_formatter(lr0_formatter)
    # ax.set_title(f"{run_name} Hyperparameter Tuning Landscape ('+' = best fitness)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Fitness")
    # # Ensure the colorbar has explicit ticks at the very bottom and top
    # # of the normalized fitness range.
    # existing_ticks = list(cbar.get_ticks())
    # eps = (z_hi - z_lo) * 1e-6 + 1e-12
    # has_z_lo = any(abs(t - z_lo) <= eps for t in existing_ticks)
    # has_z_hi = any(abs(t - z_hi) <= eps for t in existing_ticks)
    # if not has_z_lo:
    #     existing_ticks.append(z_lo)
    # if not has_z_hi:
    #     existing_ticks.append(z_hi)
    # existing_ticks = sorted(t for t in existing_ticks if z_lo - eps <= t <= z_hi + eps)

    # divide the fitness range into 5 equal parts
    cbar.set_ticks(np.linspace(z_lo, z_hi, 7))

    # Mark first max (ties -> first max only).
    best_idx = int(np.argmax(np.asarray(zs, dtype=float)))
    best_fitness = float(np.asarray(zs, dtype=float)[best_idx])
    ax.scatter(
        ys[best_idx],
        xs[best_idx],
        marker="+",
        s=220,
        c="black",
        linewidths=1.0,
        alpha=0.8,
        zorder=10,
    )
    # Label the best point value next to the "+" marker.
    # Offset in points to keep the label readable without affecting data scaling.
    ax.annotate(
        f"fitness={best_fitness:.4g}",
        xy=(ys[best_idx], xs[best_idx]),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        zorder=11,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved 2D tune plot to: {output_path}")


def _infer_run_name(csv_path: Path, runs_root: Path) -> str:
    rel = csv_path.relative_to(runs_root)
    # For paths like RUN/RUN/tune_results.csv use the top-level folder as the run name.
    name = rel.parts[0] if len(rel.parts) >= 2 else csv_path.parent.name
    name = (
        name.replace("PPL+AllPPX", "PPL+XPL-Stack")
        .replace("PPL+PPXblend", "PPL+XPL-Comp")
        .replace("PPLPPXblend", "Full-Comp")
    )
    return name


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    csv_paths = _discover_csvs(runs_root, args.glob_pattern)
    if not csv_paths:
        raise FileNotFoundError(
            f"No tune_results.csv files found under {runs_root} "
            f"with pattern '{args.glob_pattern}'"
        )
    print(f"Discovered {len(csv_paths)} run file(s).")

    z_lo, z_hi = _global_scatter_fitness_limits(csv_paths)

    _plot_fitness_vs_iterations(
        csv_paths,
        runs_root,
        args,
        runs_root / "all_runs_fitness_vs_iterations.png",
    )

    for csv_path in csv_paths:
        run_name = _infer_run_name(csv_path, runs_root)
        output_path = runs_root / f"{run_name}_tune_scatter.png"
        _plot_tune_scatter(csv_path, run_name, output_path, z_lo=z_lo, z_hi=z_hi)


if __name__ == "__main__":
    main()
