import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO_ROOT / "SLURM" / "submit_experiments.sh"
TRAIN_SCRIPT = REPO_ROOT / "SLURM" / "train_unet_multi_input.sh"
EVAL_SCRIPT = REPO_ROOT / "SLURM" / "evaluate_models_and_plot.sh"
RASTER_TO_POLYGON_SCRIPT = REPO_ROOT / "SLURM" / "raster_to_polygon.sh"


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

    def test_eval_script_uses_submit_dir_and_invokes_evaluate_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            fake_repo = temp_path / "repo"
            (fake_repo / "SLURM").mkdir(parents=True)
            (fake_repo / "src" / "training").mkdir(parents=True)

            prepare_env = fake_repo / "SLURM" / "prepare_env.sh"
            prepare_env.write_text("#!/bin/bash\n:\n", encoding="utf-8")

            spool_dir = temp_path / "spool"
            spool_dir.mkdir()
            spool_script = spool_dir / "evaluate_models_and_plot.sh"
            spool_script.write_text(
                EVAL_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
            )

            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            uv_log = temp_path / "uv_calls.txt"
            uv_path = fake_bin / "uv"
            uv_path.write_text(
                """#!/bin/bash
printf "%s\n" "$@" >> "$UV_LOG"
args=("$@")
script=""
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == *.py ]]; then
    script="${args[$i]}"
    break
  fi
done
if [[ "$script" == "../evaluation/evaluate.py" ]]; then
  output_json=""
  pred_dir=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --output-json) output_json="${args[$((i+1))]}" ;;
      --save-predictions-dir) pred_dir="${args[$((i+1))]}" ;;
    esac
  done
  mkdir -p "$(dirname "$output_json")" "$pred_dir"
  printf '{"sample":{"iou_class_1":0.5,"iou_class_2":0.2,"boundary_f1":0.3,"aji":0.4}}' > "$output_json"
  touch "$pred_dir/MWD-1#121_pred.png"
fi
exit 0
""",
                encoding="utf-8",
            )
            uv_path.chmod(
                uv_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            scratch_root = temp_path / "scratch"
            model_dir = scratch_root / "GrainSeg" / "models" / "custom"
            model_dir.mkdir(parents=True)
            (model_dir / "custom_model.keras").write_text("", encoding="utf-8")

            wheels_dir = scratch_root / "GrainSeg" / "wheels"
            wheels_dir.mkdir(parents=True)
            (
                wheels_dir / "tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"
            ).write_text("", encoding="utf-8")

            data_dir = scratch_root / "GrainSeg" / "dataset" / "sample" / "cropped"
            data_dir.mkdir(parents=True)
            (data_dir / "MWD-1#121_PPL.tif").write_text("", encoding="utf-8")
            (data_dir / "MWD-1#121_labels.tif").write_text("", encoding="utf-8")

            config_file = temp_path / "models.tsv"
            config_file.write_text(
                "Custom\tcustom_model.keras\t1\t_PPL\n",
                encoding="utf-8",
            )

            runtime_tmp = temp_path / "runtime_tmp"
            runtime_tmp.mkdir()
            output_dir = scratch_root / "GrainSeg" / "eval_trials" / "custom"

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["UV_LOG"] = str(uv_log)
            env["SLURM_SUBMIT_DIR"] = str(fake_repo)
            env["TMPDIR"] = str(runtime_tmp)
            env["SCRATCH"] = str(scratch_root)

            result = subprocess.run(
                [
                    "bash",
                    str(spool_script),
                    "--model-dir",
                    str(model_dir),
                    "--image-dir",
                    str(data_dir),
                    "--mask-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--config-file",
                    str(config_file),
                    "--ppl-image",
                    str(data_dir / "MWD-1#121_PPL.tif"),
                    "--gt-path",
                    str(data_dir / "MWD-1#121_labels.tif"),
                ],
                cwd=fake_repo,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            uv_calls = uv_log.read_text(encoding="utf-8")
            self.assertIn("sync", uv_calls)
            self.assertIn("../evaluation/evaluate.py", uv_calls)
            self.assertIn("../evaluation/plot_results.py", uv_calls)
            self.assertIn("--output-json", uv_calls)
            self.assertIn("custom_model.keras", uv_calls)
            self.assertIn("--image-suffixes", uv_calls)
            self.assertIn("_PPL", uv_calls)
            self.assertIn("quantitative_plot.png", uv_calls)
            self.assertIn("overlay.png", uv_calls)

    def test_eval_script_help_mentions_reusable_directory_and_config_inputs(
        self,
    ) -> None:
        result = subprocess.run(
            ["bash", str(EVAL_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("--model-dir", result.stdout)
        self.assertIn("--config-file", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_raster_to_polygon_script_uses_submit_dir_and_forwards_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            fake_repo = temp_path / "repo"
            (fake_repo / "SLURM").mkdir(parents=True)
            (fake_repo / "src" / "data_prep").mkdir(parents=True)

            prepare_env = fake_repo / "SLURM" / "prepare_env.sh"
            prepare_env.write_text("#!/bin/bash\n:\n", encoding="utf-8")

            spool_dir = temp_path / "spool"
            spool_dir.mkdir()
            spool_script = spool_dir / "raster_to_polygon.sh"
            spool_script.write_text(
                RASTER_TO_POLYGON_SCRIPT.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            uv_log = temp_path / "uv_calls.txt"
            uv_path = fake_bin / "uv"
            uv_path.write_text(
                """#!/bin/bash
printf "%s\n" "$@" >> "$UV_LOG"
args=("$@")
output=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --output) output="${args[$((i+1))]}" ;;
  esac
done
if [ -n "$output" ]; then
  mkdir -p "$(dirname "$output")"
  : > "$output"
fi
exit 0
""",
                encoding="utf-8",
            )
            uv_path.chmod(
                uv_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            scratch_root = temp_path / "scratch"
            input_dir = scratch_root / "GrainSeg" / "eval_trials" / "preds"
            input_dir.mkdir(parents=True)
            input_raster = input_dir / "sample_pred.png"
            input_raster.write_text("", encoding="utf-8")
            output_gpkg = scratch_root / "GrainSeg" / "eval_trials" / "sample_pred.gpkg"

            runtime_tmp = temp_path / "runtime_tmp"
            runtime_tmp.mkdir()

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["UV_LOG"] = str(uv_log)
            env["SLURM_SUBMIT_DIR"] = str(fake_repo)
            env["TMPDIR"] = str(runtime_tmp)
            env["SCRATCH"] = str(scratch_root)
            env["INPUT_RASTER"] = str(input_raster)
            env["OUTPUT_GPKG"] = str(output_gpkg)
            env["OUTPUT_LAYER"] = "predictions"
            env["CLASS_VALUE"] = "1"
            env["MIN_AREA"] = "25"
            env["NO_FLIP_Y"] = "1"

            result = subprocess.run(
                ["bash", str(spool_script)],
                cwd=fake_repo,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_gpkg.exists())

            uv_calls = uv_log.read_text(encoding="utf-8")
            self.assertIn("sync", uv_calls)
            self.assertIn("raster_to_polygon.py", uv_calls)
            self.assertIn("--input", uv_calls)
            self.assertIn("--output", uv_calls)
            self.assertIn("--output-layer", uv_calls)
            self.assertIn("predictions", uv_calls)
            self.assertIn("--class-value", uv_calls)
            self.assertIn("25", uv_calls)
            self.assertIn("--min-area", uv_calls)
            self.assertIn("--no-flip-y", uv_calls)

    def test_raster_to_polygon_script_processes_preds_directories_from_eval_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            fake_repo = temp_path / "repo"
            (fake_repo / "SLURM").mkdir(parents=True)
            (fake_repo / "src" / "data_prep").mkdir(parents=True)

            prepare_env = fake_repo / "SLURM" / "prepare_env.sh"
            prepare_env.write_text("#!/bin/bash\n:\n", encoding="utf-8")

            spool_dir = temp_path / "spool"
            spool_dir.mkdir()
            spool_script = spool_dir / "raster_to_polygon.sh"
            spool_script.write_text(
                RASTER_TO_POLYGON_SCRIPT.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            uv_log = temp_path / "uv_calls.txt"
            uv_path = fake_bin / "uv"
            uv_path.write_text(
                """#!/bin/bash
printf "%s\n" "$@" >> "$UV_LOG"
args=("$@")
output=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --output) output="${args[$((i+1))]}" ;;
  esac
done
if [ -n "$output" ]; then
  mkdir -p "$(dirname "$output")"
  : > "$output"
fi
exit 0
""",
                encoding="utf-8",
            )
            uv_path.chmod(
                uv_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

            scratch_root = temp_path / "scratch"
            eval_dir = scratch_root / "GrainSeg" / "eval" / "run_01"
            preds_ppl = eval_dir / "preds_unet_finetuned_PPL"
            preds_allppx = eval_dir / "preds_unet_finetuned_PPL+AllPPX"
            preds_ppl.mkdir(parents=True)
            preds_allppx.mkdir(parents=True)
            (preds_ppl / "MWD-1#121_pred.png").write_text("", encoding="utf-8")
            (preds_allppx / "MWD-1#121_pred.png").write_text("", encoding="utf-8")

            runtime_tmp = temp_path / "runtime_tmp"
            runtime_tmp.mkdir()

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["UV_LOG"] = str(uv_log)
            env["SLURM_SUBMIT_DIR"] = str(fake_repo)
            env["TMPDIR"] = str(runtime_tmp)
            env["SCRATCH"] = str(scratch_root)
            env["EVAL_DIR"] = str(eval_dir)
            env["CLASS_VALUE"] = "1"
            env["MIN_AREA"] = "10"

            result = subprocess.run(
                ["bash", str(spool_script)],
                cwd=fake_repo,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((preds_ppl / "MWD-1#121_pred.gpkg").exists())
            self.assertTrue((preds_allppx / "MWD-1#121_pred.gpkg").exists())

            uv_calls = uv_log.read_text(encoding="utf-8")
            self.assertIn("sync", uv_calls)
            self.assertIn("raster_to_polygon.py", uv_calls)
            self.assertIn("preds_unet_finetuned_PPL/MWD-1#121_pred.gpkg", uv_calls)
            self.assertIn(
                "preds_unet_finetuned_PPL+AllPPX/MWD-1#121_pred.gpkg",
                uv_calls,
            )


if __name__ == "__main__":
    unittest.main()
