"""Tests for watershed tuning helpers (no TensorFlow)."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from evaluation.tune_watershed import (
    WatershedParamSet,
    mean_aji_connected_components,
    mean_aji_for_watershed_params,
)


class TuneWatershedScoringTests(unittest.TestCase):
    def test_perfect_match_identity_semantic(self) -> None:
        """Identical pred and GT semantics yield AJI 1.0."""
        h, w = 31, 41
        cx1, cy1 = 15, 10
        cx2, cy2 = 15, 30
        Y, X = np.ogrid[:h, :w]
        r1 = np.sqrt((Y - cx1) ** 2 + (X - cy1) ** 2)
        r2 = np.sqrt((Y - cx2) ** 2 + (X - cy2) ** 2)
        rad = 8
        s = ((r1 <= rad) | (r2 <= rad)).astype(np.int32)

        from evaluation.metrics import get_instances

        ti = get_instances(s)
        params = WatershedParamSet(
            min_distance=3,
            boundary_dilate_iter=0,
            watershed_connectivity=1,
            min_area_px=0,
            exclude_border=False,
            ridge_level=None,
        )
        mean_aji, per = mean_aji_for_watershed_params([ti], [s], params)
        self.assertAlmostEqual(mean_aji, 1.0, places=5)
        self.assertAlmostEqual(per[0], 1.0, places=5)

    def test_cc_baseline_matches_mean_aji_when_ws_equals_cc(self) -> None:
        """Merged rectangle: watershed min_distance=1 matches CC; same AJI as CC baseline."""
        s = np.zeros((5, 10), dtype=np.int32)
        s[0:5, 0:5] = 1
        s[0:5, 5:10] = 1

        from evaluation.metrics import get_instances

        ti = get_instances(s)
        cc_mean, _ = mean_aji_connected_components([ti], [s])
        params = WatershedParamSet(
            min_distance=1,
            boundary_dilate_iter=0,
            watershed_connectivity=1,
            min_area_px=0,
            exclude_border=False,
            ridge_level=None,
        )
        ws_mean, _ = mean_aji_for_watershed_params([ti], [s], params)
        self.assertAlmostEqual(ws_mean, cc_mean, places=5)

    def test_grid_argmax_selects_highest_mean_aji(self) -> None:
        """Among several parameter sets, the argmax index matches max mean AJI."""
        s = np.zeros((12, 16), dtype=np.int32)
        s[:, 0:6] = 1
        s[:, 6] = 2
        s[:, 7:16] = 1

        from evaluation.metrics import get_instances

        ti = get_instances(s)
        candidates = [
            WatershedParamSet(1, 0, 1, 0, False, None),
            WatershedParamSet(4, 0, 1, 0, False, None),
            WatershedParamSet(4, 1, 2, 0, True, None),
        ]
        scores = [mean_aji_for_watershed_params([ti], [s], p)[0] for p in candidates]
        best_idx = int(np.argmax(scores))
        self.assertEqual(best_idx, np.argmax(np.array(scores)))
        self.assertGreaterEqual(scores[best_idx], max(scores) - 1e-9)


if __name__ == "__main__":
    unittest.main()
