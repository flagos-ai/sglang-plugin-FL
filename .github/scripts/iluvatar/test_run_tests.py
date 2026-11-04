# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Narrow command/evidence regressions; these tests never import GPU runtimes."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_PATH = Path(__file__).with_name("run_tests.py")
_SPEC = importlib.util.spec_from_file_location("_iluvatar_run_tests", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)
_CASES = [
    {"model": "qwen3_6", "case": "27b_tp4_nograph"},
    {"model": "qwen3_6", "case": "35b_a3b_tp4_nograph"},
]


def _report(command):
    return Path(
        next(
            arg.split("=", 1)[1]
            for arg in reversed(command)
            if arg.startswith("--junitxml=")
        )
    )


def _write_report(path, count, status=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    case = f"<testcase>{status}</testcase>"
    path.write_text("<testsuite>" + case * count + "</testsuite>", encoding="utf8")


class TestRunTests(unittest.TestCase):
    def setUp(self):
        # The CI caller can select D: via TEMP/TMP; all driver artifacts use this override.
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.artifacts = Path(self.folder.name)
        self.env = patch.dict(
            os.environ, {"ILUVATAR_CI_ARTIFACT_DIR": str(self.artifacts)}, clear=True
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.calls = []
        self.runner = patch.object(driver.subprocess, "run", side_effect=self.fake_run)
        self.runner.start()
        self.addCleanup(self.runner.stop)

    def fake_run(self, command, *, cwd, env):
        self.calls.append((command, cwd, dict(env)))
        if command[0] == "bash":
            output = Path(env["OUTPUT_DIR"])
            for model in ("27b", "35b_a3b"):
                for disabled in (1, 0):
                    for mode in ("offline", "concurrent"):
                        path = output / f"{model}_{mode}_graph_disabled_{disabled}.log"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("validated\n", encoding="utf8")
            for name in ("environment.log", "flagcx.log"):
                (output / name).write_text("validated\n", encoding="utf8")
        else:
            count = 1
            if "-m" in command and "not gpu" in command:
                count = 148
            elif "-c" in command:
                count = 31
            elif "serving" in command:
                count = 5
            _write_report(_report(command), count)
        return SimpleNamespace(returncode=0)

    def run_scope(self, scope, *args):
        return driver.main(["--platform", "iluvatar", "--scope", scope, *args])

    def e2e(self, task, *args):
        return self.run_scope(
            "e2e", "--task", task, "--cases", json.dumps(_CASES), *args
        )

    def test_unit_includes_example_sampling_and_preserves_minimum(self):
        self.assertEqual(self.run_scope("unit", "--", "-s"), 0)
        command, cwd, _ = self.calls[0]
        self.assertEqual(command[:3], [driver.sys.executable, "-m", "pytest"])
        self.assertEqual(
            [arg for arg in command if arg.endswith(".py")], list(driver.UNIT_TESTS)
        )
        self.assertEqual(len(driver.UNIT_TESTS), 11)
        self.assertIn(
            "tests/unit_tests/test_iluvatar_example_sampling.py", driver.UNIT_TESTS
        )
        self.assertIn("not gpu", command)
        self.assertIn("-s", command)
        self.assertEqual(cwd, str(driver.REPO_ROOT))
        self.assertEqual(_report(command), self.artifacts / "unit/contracts.xml")
        self.assertEqual(driver.MIN_UNIT_TESTS, 148)

    def test_vl_budget_default_is_independent_of_text_budget(self):
        env = {"MAX_TOKENS": "10"}
        driver._workload(env)
        self.assertEqual(env["FL_CONCURRENT_MAX_TOKENS"], "10")
        self.assertEqual(env["FL_CONCURRENT_VL_MAX_TOKENS"], "256")

    def test_explicit_vl_budget_reaches_shared_and_original_examples(self):
        os.environ["FL_CONCURRENT_VL_MAX_TOKENS"] = "1024"
        self.assertEqual(self.e2e("inference"), 0)
        self.assertTrue(any(command[0] == "bash" for command, _, _ in self.calls))
        self.assertTrue(any(command[0] != "bash" for command, _, _ in self.calls))
        for _, _, env in self.calls:
            self.assertEqual(env["FL_CONCURRENT_VL_MAX_TOKENS"], "1024")

    def test_functional_imports_sglang_before_pytest_and_selects_both_gpu_files(self):
        self.assertEqual(self.run_scope("functional"), 0)
        command = self.calls[0][0]
        self.assertEqual(command[:2], [driver.sys.executable, "-c"])
        self.assertLess(
            command[2].index("import sglang"), command[2].index("import pytest")
        )
        self.assertIn("sys.argv[1:]", command[2])
        self.assertEqual(
            [arg for arg in command if arg.endswith(".py")],
            list(driver.FUNCTIONAL_TESTS),
        )
        self.assertEqual(command[-3:-1], ["-m", "gpu"])
        self.assertEqual(driver.MIN_FUNCTIONAL_TESTS, 31)

    def test_serving_uses_original_runner_with_a_unique_report_for_each_case(self):
        self.assertEqual(self.e2e("serving", "--device", "", "--", "-s"), 0)
        self.assertEqual(len(self.calls), 2)
        reports = []
        for (command, _, env), case in zip(self.calls, _CASES):
            self.assertEqual(command[:2], [driver.sys.executable, "tests/run.py"])
            self.assertEqual(command[command.index("--device") + 1], "bi_v150")
            self.assertEqual(command[command.index("--model") + 1], case["model"])
            self.assertEqual(command[command.index("--case") + 1], case["case"])
            self.assertEqual(command[command.index("--task") + 1], "serving")
            self.assertIn("-s", command)
            self.assertEqual(env["CONCURRENT_N"], "16")
            self.assertEqual(env["MAX_TOKENS"], "256")
            self.assertEqual(env["FL_CONCURRENT_MAX_TOKENS"], "256")
            self.assertEqual(env["FL_CONCURRENT_VL_MAX_TOKENS"], "256")
            reports.append(_report(command))
        self.assertEqual(len(set(reports)), 2)

    def test_inference_also_runs_both_original_profiles_and_requires_all_sixteen_logs(
        self,
    ):
        self.assertEqual(self.e2e("inference"), 0)
        scripts = [(cmd, env) for cmd, _, env in self.calls if cmd[0] == "bash"]
        self.assertEqual(len(scripts), 2)
        self.assertEqual(
            [env["FUSED_PROFILE"] for _, env in scripts], ["vendor", "flagos"]
        )
        logs = list(self.artifacts.glob("models-*/*_graph_disabled_*.log"))
        self.assertEqual(len(logs), 16)
        for command, env in scripts:
            self.assertEqual(
                command, ["bash", "scripts/iluvatar/run_single_node_acceptance.sh"]
            )
            self.assertEqual(env["TP_SIZE"], "4")
            self.assertEqual(env["SGLANG_FL_DIST_BACKEND"], "flagcx")
            self.assertEqual(env["CONCURRENT_N"], "16")
            self.assertNotIn("SGLANG_FL_PER_OP", env)

    def test_missing_or_empty_model_logs_fail_even_when_the_script_exits_zero(self):
        for empty in (False, True):
            with self.subTest(empty=empty):

                def run(command, *, cwd, env):
                    result = self.fake_run(command, cwd=cwd, env=env)
                    if command[0] == "bash":
                        log = (
                            Path(env["OUTPUT_DIR"])
                            / "27b_concurrent_graph_disabled_0.log"
                        )
                        if empty:
                            log.write_text("", encoding="utf8")
                        else:
                            log.unlink()
                    return result

                with patch.object(driver.subprocess, "run", side_effect=run):
                    self.assertEqual(self.e2e("inference"), 1)

    def test_benchmarks_keep_the_three_original_entries_and_independent_reports(self):
        self.assertEqual(self.run_scope("benchmark"), 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(
            [cmd[cmd.index("--benchmark") + 1] for cmd, _, _ in self.calls],
            ["throughput", "latency", "serve"],
        )
        self.assertEqual(len({_report(cmd) for cmd, _, _ in self.calls}), 3)
        self.assertTrue(all(cmd[1] == "tests/run.py" for cmd, _, _ in self.calls))

    def test_minimums_skips_failures_and_collection_errors_cannot_pass(self):
        for scope, count, status in (
            ("unit", 147, ""),
            ("functional", 30, ""),
            ("functional", 31, "<skipped/>"),
            ("benchmark", 1, "<failure/>"),
            ("benchmark", 1, "<error/>"),
            ("benchmark", 0, ""),
        ):
            with self.subTest(scope=scope, count=count, status=status):

                def run(command, *, cwd, env):
                    _write_report(_report(command), count, status)
                    return SimpleNamespace(returncode=0)

                with patch.object(driver.subprocess, "run", side_effect=run):
                    self.assertEqual(self.run_scope(scope), 1)
        with patch.object(
            driver.subprocess,
            "run",
            side_effect=lambda command, **kwargs: (
                _write_report(_report(command), 4) or SimpleNamespace(returncode=0)
            ),
        ):
            self.assertEqual(self.e2e("serving"), 1)

    def test_stale_report_is_removed_and_missing_new_report_is_rejected(self):
        report = self.artifacts / "unit/contracts.xml"
        _write_report(report, 148)
        with patch.object(
            driver.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ):
            self.assertEqual(self.run_scope("unit"), 1)
        self.assertFalse(report.exists())

    def test_subprocess_failure_code_is_preserved(self):
        with patch.object(
            driver.subprocess, "run", return_value=SimpleNamespace(returncode=7)
        ) as call:
            self.assertEqual(self.run_scope("benchmark"), 7)
            self.assertEqual(call.call_count, 1)

    def test_extra_pytest_report_argument_cannot_replace_the_stage_report(self):
        self.assertEqual(
            self.run_scope(
                "benchmark", "--benchmark", "latency", "--", "--junitxml=shared.xml"
            ),
            0,
        )
        command = self.calls[0][0]
        self.assertEqual(
            command[-1], f"--junitxml={self.artifacts / 'benchmark/latency.xml'}"
        )
        self.assertIn("--junitxml=shared.xml", command)

    def test_empty_invalid_duplicate_and_unknown_cases_fail_before_execution(self):
        for cases in (
            "",
            "[]",
            "{}",
            "null",
            "[{}]",
            json.dumps(_CASES * 2),
            '[{"model":"qwen3_6","case":"missing"}]',
            '[{"model":"../escape","case":"27b"}]',
        ):
            with self.subTest(cases=cases):
                self.assertEqual(
                    self.run_scope("e2e", "--task", "serving", "--cases", cases), 1
                )
        self.assertEqual(self.run_scope("e2e", "--model", "missing"), 1)
        self.assertEqual(self.calls, [])

    def test_invalid_concurrent_workload_is_rejected_before_any_command(self):
        for name, value in (
            ("CONCURRENT_N", "0"),
            ("CONCURRENT_N", "1"),
            ("CONCURRENT_N", "-1"),
            ("MAX_TOKENS", "0"),
            ("FL_CONCURRENT_N", "0"),
            ("FL_CONCURRENT_MAX_TOKENS", "0"),
            ("FL_CONCURRENT_VL_MAX_TOKENS", "0"),
        ):
            with (
                self.subTest(name=name, value=value),
                patch.dict(os.environ, {name: value}),
            ):
                self.assertEqual(self.e2e("concurrent"), 1)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
