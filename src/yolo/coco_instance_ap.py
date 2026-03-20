"""
Build COCO-style GT/detections and run pycocotools mask AP for instance segmentation.

Used for whole held-out TIFF evaluation (SAHI predictions vs GeoPackage polygons).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from shapely.affinity import scale as scale_geometry
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient


def _iter_polygon_parts(
    geometry: Polygon | MultiPolygon | GeometryCollection,
) -> list[Polygon]:
    if geometry.is_empty:
        return []
    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return []
    if isinstance(cleaned, Polygon):
        return [cleaned]
    if isinstance(cleaned, MultiPolygon):
        return [part for part in cleaned.geoms if not part.is_empty]
    if isinstance(cleaned, GeometryCollection):
        return [
            part
            for part in cleaned.geoms
            if isinstance(part, Polygon) and not part.is_empty
        ]
    return []


def load_polygons_from_gpkg(path: Path) -> list[Polygon | MultiPolygon]:
    """Load polygon geometries from a GeoPackage (same as split_tiff_gpkg_to_yolo)."""
    geodata = gpd.read_file(path)
    polygons: list[Polygon | MultiPolygon] = []
    for geometry in geodata.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if isinstance(geometry, (Polygon, MultiPolygon)):
            polygons.append(geometry)
            continue
        if isinstance(geometry, GeometryCollection):
            polygons.extend(
                part
                for part in geometry.geoms
                if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
            )
    return polygons


def normalize_polygons_to_image_space(
    polygons: list[Polygon | MultiPolygon],
) -> list[Polygon | MultiPolygon]:
    """Flip Y if labels use negative Y (matches split_tiff_gpkg_to_yolo)."""
    if not polygons:
        return polygons
    min_y = min(p.bounds[1] for p in polygons if not p.is_empty)
    max_y = max(p.bounds[3] for p in polygons if not p.is_empty)
    if max_y <= 0 and min_y < 0:
        return [
            scale_geometry(p, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
            for p in polygons
        ]
    return polygons


def clip_polygon_to_hw(polygon: Polygon, height: int, width: int) -> list[Polygon]:
    """Clip to image rectangle [0,width] x [0,height]; return polygon parts."""
    frame = box(0, 0, width, height)
    try:
        clipped = polygon.intersection(frame)
    except Exception:
        return []
    return [p for p in _iter_polygon_parts(clipped) if not p.is_empty and p.area > 0]


def polygon_to_coco_polygon(polygon: Polygon) -> list[float]:
    """Exterior ring as flat [x0,y0,...] for COCO polygon segmentation."""
    coords = list(orient(polygon, sign=1.0).exterior.coords[:-1])
    if len(coords) < 3:
        return []
    flat: list[float] = []
    for x, y in coords:
        flat.extend((float(x), float(y)))
    return flat


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


def build_gt_annotations(
    polygons: list[Polygon | MultiPolygon],
    *,
    image_id: int,
    height: int,
    width: int,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    anns: list[dict[str, Any]] = []
    ann_id = 1
    for geom in polygons:
        for part in _iter_polygon_parts(geom):
            for clipped in clip_polygon_to_hw(part, height, width):
                seg_flat = polygon_to_coco_polygon(clipped)
                if len(seg_flat) < 6:
                    continue
                xs = seg_flat[0::2]
                ys = seg_flat[1::2]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                area = float(clipped.area)
                anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [seg_flat],
                        "area": area,
                        "bbox": [float(x0), float(y0), float(w), float(h)],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
    return anns


def object_predictions_to_coco_dt(
    predictions: list[Any],
    *,
    image_id: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    """SAHI ObjectPrediction list -> COCO detection dicts for loadRes."""
    out: list[dict[str, Any]] = []
    for pred in predictions:
        coco_p = pred.to_coco_prediction(image_id=image_id)
        record = dict(coco_p.json)
        record["image_id"] = image_id
        cid = int(record.get("category_id", 0))
        record["category_id"] = cid + 1 if cid == 0 else cid
        seg = record.get("segmentation")
        if seg is None or seg == [] or seg == {}:
            continue
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
    Uses category_id 1 by default (COCO-style); SAHI preds may use 0 -> remapped in dt builder.

    When there is no ground truth (no GT instances), summary metrics are undefined and
    reported as -1.0 (pycocotools/COCOeval convention), not as real zeros.
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
