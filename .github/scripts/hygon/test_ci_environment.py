"""CPU guards for allocation persistence, vendor preservation and CI wiring."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from ci_environment import (
    allocation_mask,
    main,
    normalized_environment,
    persist_for_actions,
    verify_image_metadata,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent


class AllocationTests(unittest.TestCase):
    def test_hip_is_authoritative_and_keeps_order(self):
        env = {
            "HIP_VISIBLE_DEVICES": "3,1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "ROCR_VISIBLE_DEVICES": "1,3",
            "UNCHANGED": "value",
        }
        selected = normalized_environment(env)
        self.assertEqual(selected["HIP_VISIBLE_DEVICES"], "3,1")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", selected)
        self.assertNotIn("ROCR_VISIBLE_DEVICES", selected)
        self.assertEqual(selected["UNCHANGED"], "value")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_unique_legacy_allocation_is_preserved_verbatim(self):
        for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            for mask in ("4,2", "", "-1", "all", "GPU-uuid-2,GPU-uuid-1"):
                with self.subTest(key=key, mask=mask):
                    selected = normalized_environment({key: mask})
                    self.assertEqual(selected["HIP_VISIBLE_DEVICES"], mask)
                    self.assertNotIn(key, selected)

    def test_equal_legacy_masks_do_not_double_filter(self):
        selected = normalized_environment(
            {"ROCR_VISIBLE_DEVICES": "3,1", "CUDA_VISIBLE_DEVICES": "3,1"}
        )
        self.assertEqual(selected["HIP_VISIBLE_DEVICES"], "3,1")
        self.assertNotIn("ROCR_VISIBLE_DEVICES", selected)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", selected)

    def test_ambiguous_legacy_remapping_is_rejected(self):
        for cuda in ("0,1", "1,3", "", "all"):
            with (
                self.subTest(cuda=cuda),
                self.assertRaisesRegex(RuntimeError, "Ambiguous"),
            ):
                allocation_mask(
                    {"ROCR_VISIBLE_DEVICES": "3,1", "CUDA_VISIBLE_DEVICES": cuda}
                )

    def test_no_allocation_does_not_invent_devices(self):
        selected = normalized_environment({})
        self.assertNotIn("HIP_VISIBLE_DEVICES", selected)
        self.assertEqual(selected["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")

    def test_empty_hip_does_not_expand_to_legacy_allocation(self):
        selected = normalized_environment(
            {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "all"}
        )
        self.assertEqual(selected["HIP_VISIBLE_DEVICES"], "")

    def test_github_env_injection_is_rejected(self):
        for value in ("0,1\nOTHER=value", "0\r", "0\0"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "single"),
            ):
                allocation_mask({"HIP_VISIBLE_DEVICES": value})

    def test_persist_only_hip_and_autoload_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "github-env"
            env = normalized_environment(
                {"ROCR_VISIBLE_DEVICES": "5,2", "GITHUB_ENV": str(path)}
            )
            persist_for_actions(env)
            text = path.read_text(encoding="utf-8")
            self.assertIn("HIP_VISIBLE_DEVICES=5,2\n", text)
            self.assertIn("TORCH_DEVICE_BACKEND_AUTOLOAD=0\n", text)
            self.assertNotIn("ROCR_VISIBLE_DEVICES", text)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", text)

    def test_real_child_gets_clean_env_and_returns_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "github-env"
            env = dict(os.environ)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            ):
                env.pop(name, None)
            env.update(
                ROCR_VISIBLE_DEVICES="6,2",
                TORCH_DEVICE_BACKEND_AUTOLOAD="1",
                GITHUB_ENV=str(path),
            )
            code = "import json,os,sys; print(json.dumps({k:os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD')})); sys.exit(7)"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "ci_environment.py"),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "HIP_VISIBLE_DEVICES": "6,2",
                    "ROCR_VISIBLE_DEVICES": None,
                    "CUDA_VISIBLE_DEVICES": None,
                    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                },
            )
            # Simulate a later Actions step retaining original container aliases.
            later = dict(env)
            later.update(
                line.split("=", 1)
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(
                normalized_environment(later)["HIP_VISIBLE_DEVICES"], "6,2"
            )

    def test_metadata_module_never_imports_torch(self):
        code = "import runpy,sys; runpy.run_path(sys.argv[1],run_name='_cpu_guard_'); assert 'torch' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code, str(SCRIPTS / "ci_environment.py")], check=True
        )


class ImageAndSetupTests(unittest.TestCase):
    def test_old_sglang_image_fails_before_dependency_queries(self):
        calls = []

        def version(package):
            calls.append(package)
            return "0.5.11"

        with self.assertRaisesRegex(RuntimeError, "0.5.11"):
            verify_image_metadata(version)
        self.assertEqual(calls, ["sglang"])

    def test_expected_image_requires_preinstalled_ci_dependencies(self):
        with self.assertRaisesRegex(RuntimeError, "missing pytest"):
            verify_image_metadata(
                lambda name: (
                    "0.5.18"
                    if name == "sglang"
                    else (_ for _ in ()).throw(RuntimeError("missing pytest"))
                )
            )

    def test_check_image_failure_cannot_execute_install_command(self):
        with (
            patch(
                "ci_environment.verify_image_metadata",
                side_effect=RuntimeError("old image"),
            ),
            patch("ci_environment.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "old image"):
                main(
                    [
                        "--check-image",
                        "--",
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "anything",
                    ]
                )
            run.assert_not_called()

    def test_setup_has_one_dependency_preserving_install(self):
        source = (SCRIPTS / "setup.sh").read_text(encoding="utf-8")
        installs = [line for line in source.splitlines() if "pip install" in line]
        self.assertEqual(len(installs), 1)
        self.assertIn("--no-deps --no-build-isolation -e", installs[0])
        self.assertNotIn("--upgrade", source)
        self.assertNotIn(".[dev]", source)
        self.assertIn("--baseline", source)
        self.assertIn("--require-gpu --min-gpus 2", source)


class WorkflowTests(unittest.TestCase):
    def test_known_runner_and_main_model_mount_are_used(self):
        config = yaml.safe_load(
            (ROOT / ".github/configs/hygon.yml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["runner_labels"], ["ci-sglang-plugin-fl-hygon"])
        self.assertIn("/data/models/Qwen:/models:ro", config["container_volumes"])
        for device in ("/dev/kfd", "/dev/mkfd", "/dev/dri"):
            self.assertIn("--device " + device, config["container_options"])
        registry = yaml.safe_load(
            (ROOT / ".github/configs/platforms.yml").read_text(encoding="utf-8")
        )
        if registry["platforms"]["hygon"]["enabled"]:
            self.assertTrue(
                config["ci_image"],
                "Hygon cannot be enabled before a real 0.5.18 image is published",
            )

    def test_standard_six_groups_and_eager_mtp_select_two_models(self):
        spec = importlib.util.spec_from_file_location(
            "hygon_ci_matrix", ROOT / ".github/scripts/generate_matrix.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.load_platform("hygon")
        matrix = module.build_e2e_matrix(config, module.get_device_sections(config), [])
        self.assertEqual(len(matrix), 7)
        self.assertEqual(
            {(x["device"], x["task"]) for x in matrix},
            {
                (device, task)
                for device in ("bw1000", "bw1000_graph")
                for task in ("inference", "serving", "concurrent")
            }
            | {("bw1000", "mtp")},
        )
        for entry in matrix:
            self.assertEqual(
                json.loads(entry["cases"]),
                [
                    {"model": "qwen3_6", "case": "27b_tp2_hygon"},
                    {"model": "qwen3_6", "case": "35b_a3b_tp2_hygon"},
                ],
            )

    def test_hygon_groups_are_serial_and_registered(self):
        workflow = (ROOT / ".github/workflows/_platform_test.yml").read_text(
            encoding="utf-8"
        )
        line = next(
            line
            for line in workflow.splitlines()
            if "max-parallel:" in line and "inputs.platform" in line
        )
        self.assertIn("inputs.platform == 'hygon'", line)
        self.assertIn("&& 1 || 256", line)
        self.assertIn(
            "test-hygon:",
            (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"),
        )

    def test_all_test_steps_clear_legacy_masks_for_hygon_only(self):
        for filename in (
            "_unit_test.yml",
            "_functional_test.yml",
            "_e2e_test.yml",
            "_benchmark_test.yml",
        ):
            workflow = yaml.safe_load(
                (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
            )
            job = next(iter(workflow["jobs"].values()))
            run = next(
                step["run"]
                for step in job["steps"]
                if "python tests/run.py" in step.get("run", "")
            )
            self.assertIn('if [ "${{ inputs.platform }}" = "hygon" ]; then', run)
            self.assertIn("unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES", run)


if __name__ == "__main__":
    unittest.main()
