"""Tests for semantic mask to instance label maps (CC and watershed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from evaluation import instance_masks


class InstanceMasksCCTests(unittest.TestCase):
    def test_two_disjoint_interior_regions(self) -> None:
        s = np.zeros((4, 6), dtype=np.int32)
        s[0:2, 0:2] = 1
        s[2:4, 4:6] = 1
        lab = instance_masks.semantic_to_instance_label_map(s)
        ids = set(np.unique(lab)) - {0}
        self.assertEqual(len(ids), 2)
        masks = dict(instance_masks.iter_instance_binary_masks(lab))
        self.assertEqual(masks[1].sum(), 4)
        self.assertEqual(masks[2].sum(), 4)

    def test_touching_interior_single_component_4_connected(self) -> None:
        """Horizontally adjacent interior (no boundary) is one 4-connected component."""
        s = np.zeros((5, 10), dtype=np.int32)
        s[0:5, 0:5] = 1
        s[0:5, 5:10] = 1
        lab = instance_masks.semantic_to_instance_label_map(s, connectivity=1)
        ids = set(np.unique(lab)) - {0}
        self.assertEqual(len(ids), 1)

    def test_min_area_drops_speckle_and_relabels(self) -> None:
        s = np.zeros((5, 5), dtype=np.int32)
        s[1:4, 1:4] = 1
        s[0, 0] = 1  # isolated pixel
        lab = instance_masks.semantic_to_instance_label_map(s, min_area_px=2)
        ids = [x for x in np.unique(lab) if x != 0]
        self.assertEqual(ids, [1])
        self.assertEqual((lab == 1).sum(), 9)


class InstanceMasksWatershedTests(unittest.TestCase):
    def test_watershed_single_peak_matches_cc_on_merged_rectangle(self) -> None:
        s = np.zeros((5, 10), dtype=np.int32)
        s[0:5, 0:5] = 1
        s[0:5, 5:10] = 1
        cc = instance_masks.semantic_to_instance_label_map(s)
        ws = instance_masks.semantic_to_instance_label_map_watershed(s, min_distance=1)
        np.testing.assert_array_equal(ws, cc)

    def test_watershed_two_disks_multiple_instances(self) -> None:
        """Euclidean interior with two lobes yields multiple distance peaks and splits."""
        h, w = 31, 41
        cx1, cy1 = 15, 10
        cx2, cy2 = 15, 30
        Y, X = np.ogrid[:h, :w]
        r1 = np.sqrt((Y - cx1) ** 2 + (X - cy1) ** 2)
        r2 = np.sqrt((Y - cx2) ** 2 + (X - cy2) ** 2)
        rad = 8
        s = ((r1 <= rad) | (r2 <= rad)).astype(np.int32)
        ws = instance_masks.semantic_to_instance_label_map_watershed(s, min_distance=5)
        n_inst = len([x for x in np.unique(ws) if x != 0])
        self.assertGreaterEqual(n_inst, 2)

    def test_watershed_boundary_column_two_lobes(self) -> None:
        """Interior separated by boundary column: interior mask has two CC; watershed labels both."""
        h, w = 12, 16
        s = np.zeros((h, w), dtype=np.int32)
        s[:, 0:6] = 1
        s[:, 6] = 2
        s[:, 7:16] = 1
        cc = instance_masks.semantic_to_instance_label_map(s)
        n_cc = len([x for x in np.unique(cc) if x != 0])
        self.assertEqual(n_cc, 2)
        ws = instance_masks.semantic_to_instance_label_map_watershed(s, min_distance=4)
        left_ids = set(np.unique(ws[:, 0:6])) - {0}
        right_ids = set(np.unique(ws[:, 7:16])) - {0}
        self.assertTrue(left_ids)
        self.assertTrue(right_ids)
        self.assertFalse(left_ids & right_ids)


class GetInstancesRegressionTests(unittest.TestCase):
    def test_get_instances_matches_semantic_helper(self) -> None:
        from evaluation import metrics

        s = np.zeros((5, 10), dtype=np.int32)
        s[0:5, 0:5] = 1
        s[0:5, 5:10] = 1
        gi = metrics.get_instances(s)
        sm = instance_masks.semantic_to_instance_label_map(s)
        np.testing.assert_array_equal(gi, sm)


if __name__ == "__main__":
    unittest.main()
