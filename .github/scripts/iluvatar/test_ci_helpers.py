# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for fail-closed CI evidence and allocated-GPU checks."""

import importlib.util
import json
import shlex
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
        cls.yaml = yaml
        cls.config = yaml.safe_load(
            (cls.repo / ".github/configs/iluvatar.yml").read_text(encoding="utf8")
        )

    def workflow(self, filename):
        return self.yaml.safe_load(
            (self.repo / ".github/workflows" / filename).read_text(encoding="utf8")
        )

    @staticmethod
    def workflow_inputs(workflow):
        # PyYAML 1.1 parses the unquoted GitHub Actions `on` key as True.
        trigger = workflow.get("on", workflow.get(True))
        return trigger["workflow_call"]["inputs"]

    @staticmethod
    def commands(workflow):
        return [
            str(step["run"])
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if "run" in step
        ]

    def test_automatic_full_resources_match_the_configured_pool(self):
        resolved = resources.resolve_resources(self.config)
        self.assertNotIn("concurrency_group", self.config)
        self.assertEqual(
            json.loads(resolved["runner_labels"]),
            ["flagcicd-iluvatar-dood-sglang518"],
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
                "runner_labels": '["special-z", "special-a", "special-z"]',
                "cuda_visible_devices": "7",
                "ci_image": "registry.example/corex:test",
            },
        )
        self.assertEqual(
            json.loads(resolved["runner_labels"]), ["special-a", "special-z"]
        )
        self.assertEqual(resolved["cuda_visible_devices"], "7")
        self.assertNotIn(":/models:ro", resolved["container_options"])

    def test_conventional_self_hosted_labels_remain_valid_when_explicit(self):
        resolved = resources.resolve_resources(
            self.config,
            overrides={"runner_labels": '["self-hosted", "custom-hardware"]'},
        )
        self.assertEqual(
            json.loads(resolved["runner_labels"]), ["custom-hardware", "self-hosted"]
        )

    def test_invalid_explicit_label_specs_are_rejected(self):
        for labels in (
            "[]",
            "{}",
            '"pool"',
            "[null]",
            '[""]',
            '[" "]',
            json.dumps(["pool\nother"]),
            json.dumps(["pool\rother"]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                resources.resolve_resources(
                    self.config, overrides={"runner_labels": labels}
                )

    def test_invalid_full_allocations_and_model_paths_fail_closed(self):
        for overrides in (
            {"cuda_visible_devices": "0"},
            {"cuda_visible_devices": "0,0,1,2"},
            {"models_path": "/model path"},
            {"models_path": "relative/models"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resources.resolve_resources(self.config, overrides=overrides)

    def test_main_ci_routes_all_hardware_through_the_shared_workflow(self):
        registry = self.yaml.safe_load(
            (self.repo / ".github/configs/platforms.yml").read_text(encoding="utf8")
        )
        workflow = self.workflow("ci.yml")
        self.assertIs(registry["platforms"]["iluvatar"]["enabled"], True)
        for platform in (
            "nvidia",
            "ascend",
            "musa",
            "thead",
            "hygon",
            "iluvatar",
        ):
            with self.subTest(platform=platform):
                job = workflow["jobs"][f"test-{platform}"]
                self.assertEqual(job["uses"], "./.github/workflows/_platform_test.yml")
                self.assertEqual(job["with"]["platform"], platform)
        self.assertIn("iluvatar-cpu-contracts", workflow["jobs"])

    def test_shared_workflows_select_a_configured_compatible_test_driver(self):
        platform = self.workflow("_platform_test.yml")
        for job_name, filename, scopes in (
            ("functional", "_functional_test.yml", ("unit", "functional")),
            ("e2e", "_e2e_test.yml", ("e2e",)),
            ("benchmark", "_benchmark_test.yml", ("benchmark",)),
        ):
            with self.subTest(job=job_name):
                self.assertEqual(
                    platform["jobs"][job_name]["uses"],
                    f"./.github/workflows/{filename}",
                )
                stage = self.workflow(filename)
                inputs = self.workflow_inputs(stage)
                self.assertEqual(inputs["test_driver"]["default"], "tests/run.py")
                commands = self.commands(stage)
                runners = [
                    command for command in commands if '"$TEST_DRIVER"' in command
                ]
                self.assertEqual(len(runners), len(scopes))
                for scope in scopes:
                    runner = next(
                        command for command in runners if f"--scope {scope}" in command
                    )
                    self.assertIn("--platform ${{ inputs.platform }}", runner)
                self.assertEqual(
                    platform["jobs"][job_name]["with"]["test_driver"],
                    "${{ fromJson(needs.setup.outputs.config).test_driver "
                    "|| 'tests/run.py' }}",
                )
        self.assertFalse(
            any(
                ".github/scripts/iluvatar/" in command
                for command in self.commands(platform)
            )
        )

    def test_iluvatar_config_preserves_strict_acceptance_in_shared_workflow(self):
        self.assertEqual(
            self.config["test_driver"], ".github/scripts/iluvatar/run_tests.py"
        )
        self.assertEqual(self.config["test_timeout"], 180)
        self.assertEqual(self.config["test_artifact_path"], "artifacts/iluvatar/")
        self.assertIs(self.config["test_artifact_required"], True)
        self.assertEqual(self.config["e2e_max_parallel"], 1)
        platform = self.workflow("_platform_test.yml")
        matrix_command = platform["jobs"]["setup"]["steps"][-1]["run"]
        self.assertIn("--ci-stage ${{ inputs.ci_stage }}", matrix_command)
        for job_name in ("functional", "e2e", "benchmark"):
            with self.subTest(job=job_name):
                values = platform["jobs"][job_name]["with"]
                self.assertIn(".test_driver", values["test_driver"])
                self.assertIn(".test_timeout", str(values["timeout"]))
                self.assertIn(".test_artifact_path", values["test_artifact_path"])
                self.assertIn(
                    ".test_artifact_required", values["test_artifact_required"]
                )
        self.assertIn(
            ".e2e_max_parallel",
            str(platform["jobs"]["e2e"]["strategy"]["max-parallel"]),
        )
        self.assertFalse((self.repo / ".github/workflows/_iluvatar_test.yml").exists())
        self.assertFalse(
            (self.repo / ".github/workflows/_iluvatar_stage_test.yml").exists()
        )
        matrix_path = self.repo / ".github/scripts/generate_matrix.py"
        spec = importlib.util.spec_from_file_location(
            "_iluvatar_matrix_contract", matrix_path
        )
        matrix_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix_module)
        manifest = self.yaml.safe_load(
            (self.repo / "tests/platforms/iluvatar.yaml").read_text(encoding="utf8")
        )
        for stage in ("pr", "nightly", "weekly"):
            with self.subTest(stage=stage):
                matrix = matrix_module.build_e2e_matrix(
                    manifest,
                    matrix_module.get_device_sections(manifest),
                    manifest["unsupported_features"],
                    ci_stage=stage,
                )
                self.assertEqual(len(matrix), 3)
                self.assertEqual(
                    {entry["task"] for entry in matrix},
                    {"inference", "concurrent", "serving"},
                )
                self.assertTrue(all(entry["device"] == "bi_v150" for entry in matrix))
                self.assertTrue(all(json.loads(entry["cases"]) for entry in matrix))

    def test_shared_stages_check_setup_run_and_upload_platform_evidence(self):
        for filename in (
            "_functional_test.yml",
            "_e2e_test.yml",
            "_benchmark_test.yml",
        ):
            with self.subTest(filename=filename):
                workflow = self.workflow(filename)
                stage = next(iter(workflow["jobs"].values()))
                steps = stage["steps"]
                commands = [str(step.get("run", "")) for step in steps]
                check = commands.index(
                    "bash .github/scripts/${{ inputs.platform }}/check.sh"
                )
                setup = commands.index(
                    "bash .github/scripts/${{ inputs.platform }}/setup.sh"
                )
                drivers = [
                    index
                    for index, command in enumerate(commands)
                    if 'python "$TEST_DRIVER"' in command
                ]
                self.assertLess(check, setup)
                self.assertTrue(drivers)
                for driver in drivers:
                    self.assertLess(setup, driver)
                    self.assertEqual(
                        steps[driver]["env"]["TEST_DRIVER"],
                        "${{ inputs.test_driver }}",
                    )
                upload = next(
                    step
                    for step in steps
                    if str(step.get("uses", "")).startswith("actions/upload-artifact@")
                )
                self.assertIn(
                    "${{ inputs.test_artifact_path }}", upload["with"]["path"]
                )
                self.assertIn(
                    "test_artifact_required", upload["with"]["if-no-files-found"]
                )
        manual = self.workflow("iluvatar-gpu.yml")["jobs"]["gpu-contracts"]
        self.assertNotIn("concurrency", manual)

    def test_driver_keeps_correctness_counts_and_original_benchmark_entries(self):
        spec = importlib.util.spec_from_file_location(
            "_iluvatar_driver_contract",
            self.repo / ".github/scripts/iluvatar/run_tests.py",
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        self.assertEqual(driver.MIN_UNIT_TESTS, 148)
        self.assertEqual(driver.MIN_FUNCTIONAL_TESTS, 31)
        self.assertEqual(len(driver.UNIT_TESTS), 11)
        self.assertIn(
            "tests/unit_tests/test_iluvatar_example_sampling.py", driver.UNIT_TESTS
        )
        self.assertEqual(
            driver.FUNCTIONAL_TESTS,
            (
                "tests/functional_tests/ops/test_iluvatar_empty_ops.py",
                "tests/unit_tests/platform/test_iluvatar_runtime.py",
            ),
        )
        self.assertEqual(driver.BENCHMARKS, ("throughput", "latency", "serve"))

    @staticmethod
    def device_env(options):
        tokens = shlex.split(options)
        values = []
        for index, token in enumerate(tokens):
            if token == "--env":
                value = tokens[index + 1]
            elif token.startswith("--env="):
                value = token.removeprefix("--env=")
            else:
                continue
            if value.startswith("CUDA_VISIBLE_DEVICES="):
                values.append(value.split("=", 1)[1])
        return values

    def test_standard_config_directly_carries_model_mount_and_gpu_env(self):
        self.assertIn(
            "/mnt/share/user_homes/models:/models:ro", self.config["container_volumes"]
        )
        self.assertEqual(self.device_env(self.config["container_options"]), ["0,1,2,3"])
        resolved = resources.resolve_resources(self.config)
        self.assertEqual(
            resolved["container_options"].count(
                "/mnt/share/user_homes/models:/models:ro"
            ),
            1,
        )

    def test_manual_model_mount_replaces_the_standard_source_once(self):
        resolved = resources.resolve_resources(
            self.config, overrides={"models_path": "/srv/ci-models"}
        )
        mounts = [
            token.removeprefix("--volume=")
            for token in shlex.split(resolved["container_options"])
            if token.startswith("--volume=") and ":/models:" in token
        ]
        self.assertEqual(mounts, ["/srv/ci-models:/models:ro"])

    def test_operators_remove_models_and_manual_gpu_env_replaces_old_allocation(self):
        resolved = resources.resolve_resources(
            self.config, scope="operators", overrides={"cuda_visible_devices": "7"}
        )
        self.assertNotIn(":/models", resolved["container_options"])
        self.assertEqual(self.device_env(resolved["container_options"]), ["7"])


if __name__ == "__main__":
    unittest.main()
