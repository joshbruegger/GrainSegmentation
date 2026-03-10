import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO_ROOT / "SLURM" / "submit_experiments.sh"
TRAIN_SCRIPT = REPO_ROOT / "SLURM" / "train_unet_multi_input.sh"


class SlurmScriptTests(unittest.TestCase):
    def test_train_script_uses_submit_dir_when_running_from_spool_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            fake_repo = temp_path / "repo"
            fake_repo.mkdir()
            (fake_repo / "SLURM").mkdir()
            (fake_repo / "src" / "training").mkdir(parents=True)

            prepare_env = fake_repo / "SLURM" / "prepare_env.sh"
            prepare_env.write_text("#!/bin/bash\n:\n", encoding="utf-8")

            spool_dir = temp_path / "spool"
            spool_dir.mkdir()
            spool_script = spool_dir / "train_unet_multi_input.sh"
            spool_script.write_text(
                TRAIN_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
            )

            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            uv_log = temp_path / "uv_calls.txt"
            uv_path = fake_bin / "uv"
            uv_path.write_text(
                '#!/bin/bash\nprintf "%s\n" "$@" >> "$UV_LOG"\nexit 0\n',
                encoding="utf-8",
            )
            uv_path.chmod(
                uv_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            scratch_root = temp_path / "scratch"
            dataset_dir = (
                scratch_root / "GrainSeg" / "dataset" / "MWD-1#121" / "cropped"
            )
            dataset_dir.mkdir(parents=True)
            runtime_tmp = temp_path / "runtime_tmp"
            runtime_tmp.mkdir()

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["UV_LOG"] = str(uv_log)
            env["SLURM_SUBMIT_DIR"] = str(fake_repo)
            env["TMPDIR"] = str(runtime_tmp)
            env["SCRATCH"] = str(scratch_root)

            result = subprocess.run(
                ["bash", str(spool_script), "--verbose"],
                cwd=fake_repo,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("sync", uv_log.read_text(encoding="utf-8"))
            self.assertIn("--validation-fraction", uv_log.read_text(encoding="utf-8"))

    def test_submit_experiments_forwards_verbose_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            sbatch_args_log = temp_path / "sbatch_args.txt"

            sbatch_path = fake_bin / "sbatch"
            sbatch_path.write_text(
                '#!/bin/bash\nprintf "%s\n" "$@" > "$SBATCH_ARGS_LOG"\n',
                encoding="utf-8",
            )
            sbatch_path.chmod(
                sbatch_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["SBATCH_ARGS_LOG"] = str(sbatch_args_log)

            result = subprocess.run(
                ["bash", str(SUBMIT_SCRIPT), "--ppl", "--verbose"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--verbose", sbatch_args_log.read_text(encoding="utf-8"))

    def test_train_script_help_mentions_verbose_flag(self) -> None:
        result = subprocess.run(
            ["bash", str(TRAIN_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("--verbose", result.stdout)
        self.assertIn("validation holdout", result.stdout)


if __name__ == "__main__":
    unittest.main()
