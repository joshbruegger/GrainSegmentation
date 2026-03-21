#!/usr/bin/env python3
"""
Emit one argv token per line for evaluate.py from tune_watershed --output-json.

Reads best_params (same schema as tune_watershed.py) and prints flags matching
semantic_to_instance_label_map_watershed / evaluate.py watershed CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: watershed_json_to_eval_args.py <watershed_best.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    bp = payload.get("best_params")
    if not isinstance(bp, dict):
        print("JSON missing best_params object", file=sys.stderr)
        sys.exit(1)

    min_distance = int(bp["min_distance"])
    boundary_dilate_iter = int(bp["boundary_dilate_iter"])
    watershed_connectivity = int(bp["watershed_connectivity"])
    min_area_px = int(bp["min_area_px"])
    exclude_border = bool(bp["exclude_border"])
    ridge_level = bp.get("ridge_level")

    lines: list[str] = [
        "--instance-method",
        "watershed",
        "--watershed-min-distance",
        str(min_distance),
        "--watershed-boundary-dilate-iter",
        str(boundary_dilate_iter),
        "--watershed-connectivity",
        str(watershed_connectivity),
        "--watershed-min-area-px",
        str(min_area_px),
    ]
    if exclude_border:
        lines.append("--watershed-exclude-border")
    else:
        lines.append("--no-watershed-exclude-border")
    if ridge_level is not None:
        lines.extend(["--watershed-ridge-level", str(float(ridge_level))])

    for token in lines:
        print(token)


if __name__ == "__main__":
    main()
