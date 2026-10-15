# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for fail-closed CI evidence and allocated-GPU checks."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
