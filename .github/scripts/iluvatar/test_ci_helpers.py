# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for fail-closed CI evidence and allocated-GPU checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "iluvatar"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


results = _load("check_ci_results")
idle = _load("require_idle_gpus")
resources = _load("resolve_ci_resources")


class TestCiEvidence(unittest.TestCase):
    def check_report(self, content, minimum=1):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.xml"
            path.write_text(content, encoding="utf8")
            return results.check_results(path, minimum)

    def test_executed_passing_cases_are_accepted(self):
        self.assertEqual(
            self.check_report(
                '<testsuites><testsuite><testcase name="ran"/></testsuite></testsuites>'
            ),
            1,
        )

    def test_skip_failure_and_collection_error_are_rejected(self):
        for status in ("skipped", "failure", "error"):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                self.check_report(
                    f"<testsuite><testcase><{status}/></testcase></testsuite>"
                )

    def test_empty_or_partial_suites_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_report('<testsuite tests="31"/>', 31)
        with self.assertRaises(RuntimeError):
            self.check_report("<testsuite><testcase/></testsuite>", 31)

    def test_missing_report_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaises(FileNotFoundError),
        ):
            results.check_results(Path(folder) / "missing.xml")


class TestIdleAllocation(unittest.TestCase):
    inventory = "index, memory.used [MiB], memory.total [MiB]\n0, 24170 MiB, 32768 MiB\n3, 0 MiB, 32768 MiB\n7, 32 MiB, 32768 MiB\n"

    def test_only_explicit_idle_devices_are_accepted(self):
        checked = idle.check_inventory(self.inventory, "3,7")
        self.assertEqual([gpu["index"] for gpu in checked], [3, 7])

    def test_existing_workload_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "busy memory"):
            idle.check_inventory(self.inventory, "0")

    def test_absent_device_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "absent"):
            idle.check_inventory(self.inventory, "15")

    def test_implicit_duplicate_and_invalid_allocations_are_rejected(self):
        for selected in ("", "all", "3,3", "-1", "3;7"):
            with self.subTest(selected=selected), self.assertRaises(RuntimeError):
                idle.check_inventory(self.inventory, selected)

    def test_unusable_inventory_fails_closed(self):
        for text in (
            "",
            "GPU 3 is idle",
            "index, memory.used, memory.total\n3, N/A, 32768 MiB",
            "index, memory.used, memory.total\n3, 0, 0",
        ):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                idle.check_inventory(text, "3")


class TestConfiguredResources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        cls.repo = _SCRIPTS.parents[1]
        cls.config = yaml.safe_load(
            (cls.repo / ".github/configs/iluvatar.yml").read_text(encoding="utf8")
        )

    def test_automatic_full_resources_match_the_configured_pool(self):
        resolved = resources.resolve_resources(self.config)
        self.assertEqual(
            json.loads(resolved["runner_labels"]),
            ["iluvatar-cicd-vllm-plugin", "self-hosted"],
        )
        self.assertEqual(resolved["cuda_visible_devices"], "0,1,2,3")
        self.assertTrue(
            resolved["ci_image"].startswith("harbor.baai.ac.cn/plugin/sglang-plugin-fl")
        )
        for mount in (
            "/dev:/dev",
            "/lib/modules:/lib/modules",
            "/usr/local/corex-4.5.0/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1",
            "/mnt/share/user_homes/models:/models:ro",
        ):
            self.assertIn(f"--volume={mount}", resolved["container_options"])

    def test_manual_overrides_are_explicit_and_canonical(self):
        resolved = resources.resolve_resources(
            self.config,
            scope="operators",
            overrides={
                "runner_labels": '["self-hosted", "special", "special"]',
                "cuda_visible_devices": "7",
                "ci_image": "registry.example/corex:test",
            },
        )
        self.assertEqual(
            json.loads(resolved["runner_labels"]), ["self-hosted", "special"]
        )
        self.assertEqual(resolved["cuda_visible_devices"], "7")
        self.assertNotIn(":/models:ro", resolved["container_options"])

    def test_invalid_full_allocations_and_model_paths_fail_closed(self):
        for overrides in (
            {"cuda_visible_devices": "0"},
            {"cuda_visible_devices": "0,0,1,2"},
            {"models_path": "/model path"},
            {"models_path": "relative/models"},
            {"runner_labels": '["unknown-without-self-hosted"]'},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resources.resolve_resources(self.config, overrides=overrides)

    def test_main_ci_actually_enables_full_iluvatar_acceptance(self):
        import yaml

        registry = yaml.safe_load(
            (self.repo / ".github/configs/platforms.yml").read_text(encoding="utf8")
        )
        workflow = yaml.safe_load(
            (self.repo / ".github/workflows/ci.yml").read_text(encoding="utf8")
        )
        self.assertIs(registry["platforms"]["iluvatar"]["enabled"], True)
        job = workflow["jobs"]["test-iluvatar"]
        self.assertEqual(job["uses"], "./.github/workflows/iluvatar-gpu.yml")
        self.assertEqual(job["with"]["scope"], "full")


if __name__ == "__main__":
    unittest.main()
