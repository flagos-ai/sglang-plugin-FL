# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for vendor-stack validation and Hygon test selection."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
spec = importlib.util.spec_from_file_location(
    "hygon_verify_environment", Path(__file__).with_name("verify_environment.py")
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

TestRunner = importlib.import_module("tests.run").TestRunner
ModelConfig = __import__(
    "tests.utils.model_config", fromlist=["ModelConfig"]
).ModelConfig


class EnvironmentVerifierTests(unittest.TestCase):
    def setUp(self):
        print_patch = patch("builtins.print")
        print_patch.start()
        self.addCleanup(print_patch.stop)
        self.versions = {
            "sglang": "0.5.18",
            "torch": "2.10.0+vendor.dtk.build",
            "sglang-kernel": "0.4.2.post2+vendor.torch2100",
            "flagtree": "0.6.1+hcu3.6",
            "flag-gems": "5.4.0.dev0",
        }
        self.version_patch = patch.object(
            verifier.metadata, "version", side_effect=self.package_version
        )
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    def package_version(self, name):
        if name not in self.versions:
            raise metadata.PackageNotFoundError(name)
        return self.versions[name]

    def test_flat_docker_copy_can_capture_base_versions(self):
        source_path = Path(__file__).with_name("verify_environment.py")
        namespace = {
            "__file__": "/opt/hygon-verify-environment.py",
            "__name__": "flat_hygon_verifier",
        }
        exec(
            compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"),
            namespace,
        )
        self.assertEqual(namespace["REPO_ROOT"], Path.cwd())
        with patch.object(Path, "write_text", autospec=True) as writer:
            rc = namespace["main"](["--capture-vendor-stack", "/opt/baseline.json"])
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(writer.call_args_list[0].args[1])["torch"],
            self.versions["torch"],
        )

    def test_local_dtk_versions_are_preserved_without_guessing_tag_metadata(self):
        with patch.object(Path, "write_text", autospec=True) as writer:
            verifier.capture_vendor_stack(Path("/opt/baseline.json"))
        serialized = writer.call_args_list[0].args[1]
        constraints = writer.call_args_list[1].args[1]
        self.assertEqual(json.loads(serialized)["torch"], self.versions["torch"])
        self.assertIn("torch==2.10.0+vendor.dtk.build", constraints)
        self.assertIn("sglang-kernel==0.4.2.post2+vendor.torch2100", constraints)

    def test_baseline_detects_vendor_wheel_replacement(self):
        baseline = {
            name: self.versions[name] for name in verifier.REQUIRED_VENDOR_DISTRIBUTIONS
        }
        self.versions["torch"] = "2.10.0+replaced.vendor.build"
        with patch.object(Path, "read_text", return_value=json.dumps(baseline)):
            with self.assertRaisesRegex(RuntimeError, "vendor stack changed.*torch"):
                verifier._verify_versions(Path("/opt/baseline.json"))

    def test_missing_baseline_entry_cannot_silently_pass(self):
        with patch.object(Path, "read_text", return_value='{"torch": "2.10.0"}'):
            with self.assertRaisesRegex(RuntimeError, "missing distributions"):
                verifier._verify_versions(Path("/opt/baseline.json"))

    def test_wrong_sglang_release_fails(self):
        self.versions["sglang"] = "0.5.11"
        with self.assertRaisesRegex(RuntimeError, "sglang must have release 0.5.18"):
            verifier._verify_versions(None)

    def test_missing_vendor_distribution_has_clear_diagnostic(self):
        del self.versions["sglang-kernel"]
        with self.assertRaisesRegex(RuntimeError, "missing: sglang-kernel"):
            verifier._verify_versions(None)

    def test_cuda_only_torch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Hygon HCU/HIP"):
            verifier._verify_hygon_torch(
                SimpleNamespace(version=SimpleNamespace(hip=None))
            )

    def test_hcu_and_hip_runtime_markers_are_accepted(self):
        verifier._verify_hygon_torch(
            SimpleNamespace(__hcu_version__="3.6", version=SimpleNamespace(hip=None))
        )
        verifier._verify_hygon_torch(
            SimpleNamespace(version=SimpleNamespace(hip="6.4"))
        )

    def test_shadowing_old_plugin_copy_is_rejected(self):
        with patch.object(
            verifier.importlib.util,
            "find_spec",
            return_value=SimpleNamespace(origin=str(REPO_ROOT / "old" / "__init__.py")),
        ):
            with self.assertRaisesRegex(RuntimeError, "older plugin copy"):
                verifier._verify_plugin_path(REPO_ROOT)

    def test_metadata_only_does_not_load_gpu_libraries(self):
        with (
            patch.object(verifier.sys, "version_info", (3, 10, 12)),
            patch.object(verifier, "_verify_plugin_path"),
            patch.object(verifier, "_require_import") as importer,
        ):
            verifier.verify(metadata_only=True)
        importer.assert_not_called()

    def test_missing_python_runtime_dependency_names_the_failed_import(self):
        with patch.object(
            verifier.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'sqlalchemy'"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "runtime import failed: sqlalchemy"
            ):
                verifier._require_import("sqlalchemy")

    def test_missing_flagcx_path_fails_without_native_fallback(self):
        with patch.dict(verifier.os.environ, {"FLAGCX_PATH": ""}):
            with self.assertRaisesRegex(RuntimeError, "FLAGCX_PATH"):
                verifier._verify_flagcx()

    def test_insufficient_gpus_fail_before_loading_kernels(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )
        with self.assertRaisesRegex(RuntimeError, "at least 2 Hygon GPUs"):
            verifier._verify_gpu(torch, 2)

    def test_metadata_only_and_gpu_check_cannot_be_combined(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            verifier.verify(metadata_only=True, require_gpu=True)


class HygonAcceptanceSelectionTests(unittest.TestCase):
    def test_eager_and_graph_select_both_models_and_all_three_tasks(self):
        expected_pairs = {
            (task, case)
            for task in ("inference", "serving", "concurrent")
            for case in ("27b_tp2_hygon", "35b_a3b_tp2_hygon")
        }
        for device, graph in (("bw1000", False), ("bw1000_graph", True)):
            with self.subTest(device=device):
                runner = TestRunner("hygon", device, "e2e")
                selected = runner.discover_tests()
                self.assertEqual(
                    {(item.name.split("/")[1], item.case) for item in selected},
                    expected_pairs,
                )
                for item in selected:
                    config = ModelConfig.load(item.model, item.case)
                    config.apply_engine_overrides(item.engine_overrides)
                    kwargs = config.engine_kwargs()
                    self.assertEqual(kwargs["tp_size"], 2)
                    self.assertEqual(kwargs["pp_size"], 1)
                    self.assertEqual(kwargs["attention_backend"], "hcu")
                    self.assertEqual(kwargs["dtype"], "bfloat16")
                    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
                    self.assertEqual(kwargs["page_size"], 64)
                    self.assertTrue(kwargs["disable_radix_cache"])
                    self.assertEqual(kwargs["disable_cuda_graph"], not graph)
                    self.assertEqual(kwargs["cuda_graph_backend_prefill"], "disabled")
                    self.assertNotIn("enable_breakable_cuda_graph", kwargs)
                    self.assertNotIn("cuda_graph_max_bs", kwargs)
                    if graph:
                        self.assertEqual(kwargs["cuda_graph_backend_decode"], "full")
                        self.assertEqual(kwargs["cuda_graph_max_bs_decode"], 16)

    def test_hardware_acceptance_does_not_enable_unpublished_ci_platform(self):
        import yaml

        registry = yaml.safe_load(
            (REPO_ROOT / ".github/configs/platforms.yml").read_text(encoding="utf-8")
        )
        self.assertFalse(registry["platforms"].get("hygon", {}).get("enabled", False))
        self.assertEqual(
            TestRunner("hygon", "bw1000", "benchmark").discover_tests(), []
        )


if __name__ == "__main__":
    unittest.main()
