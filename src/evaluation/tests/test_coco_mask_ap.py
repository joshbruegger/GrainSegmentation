"""Tests for COCO mask AP helpers (raster instance maps)."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from evaluation.coco_mask_ap import (
    aggregate_coco_means,
    evaluate_mask_ap,
    instance_label_map_to_coco_dt,
    instance_label_map_to_coco_gt,
)


class CocoMaskApAdapterTests(unittest.TestCase):
    def test_gt_and_dt_perfect_match_high_ap(self) -> None:
        lab = np.zeros((32, 32), dtype=np.int32)
        lab[4:20, 4:20] = 1
        gt = instance_label_map_to_coco_gt(lab, image_id=1, height=32, width=32)
        dt = instance_label_map_to_coco_dt(lab, image_id=1, height=32, width=32)
        self.assertEqual(len(gt), 1)
        self.assertEqual(len(dt), 1)
        self.assertIn("segmentation", gt[0])
        self.assertIn("segmentation", dt[0])
        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.png",
            height=32,
            width=32,
            gt_annotations=gt,
            dt_annotations=dt,
        )
        self.assertGreater(summary.ap_50, 0.99)
        self.assertGreater(summary.ap_50_95, 0.99)

    def test_empty_dt_zero_ap(self) -> None:
        lab = np.zeros((16, 16), dtype=np.int32)
        lab[2:8, 2:8] = 1
        gt = instance_label_map_to_coco_gt(lab, image_id=1, height=16, width=16)
        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.png",
            height=16,
            width=16,
            gt_annotations=gt,
            dt_annotations=[],
        )
        self.assertEqual(summary.ap_50, 0.0)
        self.assertEqual(summary.ap_50_95, 0.0)

    def test_no_gt_sentinels(self) -> None:
        summary = evaluate_mask_ap(
            image_id=1,
            file_name="t.png",
            height=10,
            width=10,
            gt_annotations=[],
            dt_annotations=[],
        )
        self.assertEqual(summary.ap_50_95, -1.0)
        self.assertEqual(summary.ap_50, -1.0)

    def test_aggregate_coco_means_skips_negative(self) -> None:
        rows = [
            {
                "AP": 0.5,
                "AP50": 0.6,
                "AP75": 0.4,
                "APs": -1.0,
                "APm": -1.0,
                "APl": -1.0,
                "AR1": 0.1,
                "AR10": 0.2,
                "AR100": 0.3,
            },
            {
                "AP": 0.7,
                "AP50": 0.8,
                "AP75": 0.6,
                "APs": -1.0,
                "APm": -1.0,
                "APl": -1.0,
                "AR1": 0.2,
                "AR10": 0.3,
                "AR100": 0.4,
            },
        ]
        m = aggregate_coco_means(rows)
        self.assertAlmostEqual(m["mean_AP"], 0.6)
        self.assertAlmostEqual(m["mean_AP50"], 0.7)


if __name__ == "__main__":
    unittest.main()
