"""
COCO mask AP from 2D instance label maps (numpy), aligned with YOLO's coco_instance_ap.

Converts raster instance IDs to COCO RLE segmentations and runs pycocotools COCOeval.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _ensure_dt_bbox_area(record: dict[str, Any], *, height: int, width: int) -> None:
    if record.get("bbox") and record.get("area"):
        return
    seg = record.get("segmentation")
    if seg is None or seg == [] or seg == {}:
        return
    if isinstance(seg, dict):
        record["area"] = float(mask_utils.area(seg))
        record["bbox"] = [float(x) for x in mask_utils.toBbox(seg)]
        return
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = (
        mask_utils.merge(rles)
        if isinstance(rles, list) and len(rles) > 1
        else (rles[0] if isinstance(rles, list) else rles)
    )
    record["area"] = float(mask_utils.area(rle))
    record["bbox"] = [float(x) for x in mask_utils.toBbox(rle)]


def _binary_mask_to_rle(mask_hw: np.ndarray) -> dict[str, Any]:
    """Encode a single (H,W) boolean or {0,1} mask as COCO RLE dict."""
    m = np.asfortranarray(mask_hw.astype(np.uint8))
    return mask_utils.encode(m)


def instance_label_map_to_coco_gt(
    instance_label_map: np.ndarray,
    *,
    image_id: int,
    height: int,
    width: int,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    """
    Ground-truth COCO annotations from an instance label map (0 = background).

    Each connected instance becomes one polygon-free RLE annotation with ``iscrowd=0``.
    """
    if instance_label_map.shape != (height, width):
        raise ValueError(
            f"instance_label_map shape {instance_label_map.shape} != ({height}, {width})"
        )
    anns: list[dict[str, Any]] = []
    ann_id = 1
    for lid in sorted(x for x in np.unique(instance_label_map) if x != 0):
        binary = instance_label_map == lid
        if not np.any(binary):
            continue
        rle = _binary_mask_to_rle(binary)
        record: dict[str, Any] = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "iscrowd": 0,
        }
        _ensure_dt_bbox_area(record, height=height, width=width)
        anns.append(record)
        ann_id += 1
    return anns


def instance_label_map_to_coco_dt(
    instance_label_map: np.ndarray,
    *,
    image_id: int,
    height: int,
    width: int,
    category_id: int = 1,
    score: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Detection list for COCO ``loadRes`` from a predicted instance label map.

    Each instance gets ``score=1.0`` by default (no confidence from semantic post-process).
    """
    if instance_label_map.shape != (height, width):
        raise ValueError(
            f"instance_label_map shape {instance_label_map.shape} != ({height}, {width})"
        )
    out: list[dict[str, Any]] = []
    for lid in sorted(x for x in np.unique(instance_label_map) if x != 0):
        binary = instance_label_map == lid
        if not np.any(binary):
            continue
        rle = _binary_mask_to_rle(binary)
        record: dict[str, Any] = {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "score": float(score),
        }
        _ensure_dt_bbox_area(record, height=height, width=width)
        out.append(record)
    return out


@dataclass
class InstanceAPSummary:
    ap_50_95: float
    ap_50: float
    ap_75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar_1: float
    ar_10: float
    ar_100: float
    raw_stats: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "AP": float(self.ap_50_95),
            "AP50": float(self.ap_50),
            "AP75": float(self.ap_75),
            "APs": float(self.ap_small),
            "APm": float(self.ap_medium),
            "APl": float(self.ap_large),
            "AR1": float(self.ar_1),
            "AR10": float(self.ar_10),
            "AR100": float(self.ar_100),
        }
        if self.raw_stats is not None:
            d["coco_stats"] = self.raw_stats.tolist()
        return d


def evaluate_mask_ap(
    *,
    image_id: int,
    file_name: str,
    height: int,
    width: int,
    gt_annotations: list[dict[str, Any]],
    dt_annotations: list[dict[str, Any]],
    category_id: int = 1,
    category_name: str = "grain",
) -> InstanceAPSummary:
    """
    Run COCO mask evaluation for a single image (image_id must match GT/DT).

    When there is no ground truth (no GT instances), summary metrics are undefined and
    reported as -1.0 (pycocotools/COCOeval convention).
    """
    if not gt_annotations:
        return InstanceAPSummary(
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            None,
        )

    categories = [{"id": category_id, "name": category_name}]
    images = [
        {"id": image_id, "width": width, "height": height, "file_name": file_name}
    ]
    dataset = {
        "images": images,
        "annotations": gt_annotations,
        "categories": categories,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()

    if not dt_annotations:
        return InstanceAPSummary(
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            None,
        )

    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        coco_dt = coco_gt.loadRes(dt_annotations)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    return InstanceAPSummary(
        float(stats[0]),
        float(stats[1]),
        float(stats[2]),
        float(stats[3]),
        float(stats[4]),
        float(stats[5]),
        float(stats[6]),
        float(stats[7]),
        float(stats[8]),
        stats,
    )


def aggregate_coco_means(
    per_image: list[dict[str, Any]],
) -> dict[str, float | None]:
    """
    Mean of per-image COCO summary fields. Excludes undefined sentinels (-1) and NaNs.
    Same rule as YOLO's aggregate_sahi_means.
    """
    mean_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
    )
    out: dict[str, float | None] = {}
    for key in mean_keys:
        values = [
            float(row[key])
            for row in per_image
            if np.isfinite(row[key]) and row[key] >= 0
        ]
        out[f"mean_{key}"] = float(np.mean(values)) if values else None
    return out
