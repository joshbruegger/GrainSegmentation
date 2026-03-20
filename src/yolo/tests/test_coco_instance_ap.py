import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock


REPO_YOLO = Path(__file__).resolve().parents[1]
if str(REPO_YOLO) not in sys.path:
    sys.path.insert(0, str(REPO_YOLO))


class CocoInstanceApTests(unittest.TestCase):
    def test_build_gt_and_perfect_ap(self) -> None:
        from shapely.geometry import Polygon

        from coco_instance_ap import build_gt_annotations, evaluate_mask_ap

        p = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])
        gt = build_gt_annotations([p], image_id=1, height=100, width=100)
        self.assertEqual(len(gt), 1)
        a = gt[0]
        dt = [
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": a["segmentation"],
                "score": 0.99,
                "bbox": a["bbox"],
                "area": a["area"],
            }
        ]
        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.tif",
            height=100,
            width=100,
            gt_annotations=gt,
            dt_annotations=dt,
        )
        self.assertGreater(summary.ap_50, 0.99)
        self.assertGreater(summary.ap_50_95, 0.99)

    def test_empty_predictions_ap_zero(self) -> None:
        from shapely.geometry import Polygon

        from coco_instance_ap import build_gt_annotations, evaluate_mask_ap

        p = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        gt = build_gt_annotations([p], image_id=1, height=100, width=100)
        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.tif",
            height=100,
            width=100,
            gt_annotations=gt,
            dt_annotations=[],
        )
        self.assertEqual(summary.ap_50, 0.0)
        self.assertEqual(summary.ap_50_95, 0.0)

    def test_no_gt_returns_undefined_sentinels(self) -> None:
        from coco_instance_ap import evaluate_mask_ap

        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.tif",
            height=100,
            width=100,
            gt_annotations=[],
            dt_annotations=[],
        )
        # pycocotools-style: no GT => summary metrics undefined (-1), not real zeros
        self.assertEqual(summary.ap_50_95, -1.0)
        self.assertEqual(summary.ap_50, -1.0)
        self.assertEqual(summary.ap_75, -1.0)

    def test_object_predictions_to_coco_dt_fills_bbox(self) -> None:
        from coco_instance_ap import object_predictions_to_coco_dt

        pred = MagicMock()
        coco_p = MagicMock()
        coco_p.json = {
            "bbox": None,
            "score": 0.9,
            "category_id": 0,
            "category_name": "grain",
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "iscrowd": 0,
            "area": None,
        }
        pred.to_coco_prediction.return_value = coco_p
        dt = object_predictions_to_coco_dt([pred], image_id=1, height=64, width=64)
        self.assertEqual(len(dt), 1)
        self.assertEqual(dt[0]["category_id"], 1)
        self.assertIn("bbox", dt[0])
        self.assertGreater(dt[0]["area"], 0)


if __name__ == "__main__":
    unittest.main()
