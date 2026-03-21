"""Tests for watershed_json_to_eval_args CLI helper."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_SRC = Path(__file__).resolve().parents[2]
HELPER = REPO_SRC / "evaluation" / "watershed_json_to_eval_args.py"


class WatershedJsonToEvalArgsTests(unittest.TestCase):
    def test_emits_flags_for_full_best_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ws.json"
            path.write_text(
                json.dumps(
                    {
                        "best_params": {
                            "min_distance": 3,
                            "boundary_dilate_iter": 1,
                            "watershed_connectivity": 2,
                            "min_area_px": 5,
                            "exclude_border": True,
                            "ridge_level": 2.5,
                        }
                    }
                ),
                encoding="utf-8",
            )
            out = subprocess.check_output(
                [sys.executable, str(HELPER), str(path)],
                text=True,
            )
            tokens = out.strip().split("\n")
            self.assertIn("--instance-method", tokens)
            self.assertIn("watershed", tokens)
            self.assertEqual(
                tokens,
                [
                    "--instance-method",
                    "watershed",
                    "--watershed-min-distance",
                    "3",
                    "--watershed-boundary-dilate-iter",
                    "1",
                    "--watershed-connectivity",
                    "2",
                    "--watershed-min-area-px",
                    "5",
                    "--watershed-exclude-border",
                    "--watershed-ridge-level",
                    "2.5",
                ],
            )

    def test_omits_ridge_level_when_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ws.json"
            path.write_text(
                json.dumps(
                    {
                        "best_params": {
                            "min_distance": 1,
                            "boundary_dilate_iter": 0,
                            "watershed_connectivity": 1,
                            "min_area_px": 0,
                            "exclude_border": False,
                            "ridge_level": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            out = subprocess.check_output(
                [sys.executable, str(HELPER), str(path)],
                text=True,
            )
            self.assertNotIn("--watershed-ridge-level", out)
            self.assertIn("--no-watershed-exclude-border", out)


if __name__ == "__main__":
    unittest.main()
