"""Stdlib-only checks for platform test dispatch and process exit status."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("run_platform_tests.py")
SPEC = importlib.util.spec_from_file_location("run_platform_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DISPATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCHER)


class PlatformDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_runner(self, relative_path, target, status):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import json, sys\n"
            f"print(json.dumps({{'target': {target!r}, 'argv': sys.argv[1:]}}))\n"
            f"raise SystemExit({status})\n",
            encoding="utf-8",
        )
        return path

    def run_dispatcher(self, args):
        script = self.root / ".github/scripts/run_platform_tests.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT, script)
        return subprocess.run(
            [sys.executable, "-B", str(script), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_generic_runner_receives_original_arguments(self):
        args = ["--platform", "cuda", "--scope", "unit", "--", "-k", "real case"]
        command = DISPATCHER.build_command(args, self.root)
        self.assertEqual(
            command, [sys.executable, str(self.root / "tests/run.py"), *args]
        )

    def test_existing_platform_hook_receives_every_argument(self):
        hook = self.write_runner(".github/scripts/iluvatar/run_tests.py", "hook", 0)
        args = [
            "--platform=iluvatar",
            "--scope",
            "e2e",
            "--task",
            "serving",
            "--device",
            "bi_v150",
            "--cases",
            '[{"model":"qwen3_6","case":"27b_tp4_nograph"}]',
            "--future-option",
            "value",
            "--",
            "--junitxml=artifacts/result.xml",
            "-k",
            "real case",
        ]
        self.assertEqual(
            DISPATCHER.build_command(args, self.root),
            [sys.executable, str(hook), *args],
        )

    def test_platform_like_pytest_argument_does_not_select_hook(self):
        self.write_runner(".github/scripts/iluvatar/run_tests.py", "hook", 0)
        args = ["--platform", "cuda", "--", "--platform", "iluvatar"]
        self.assertEqual(
            DISPATCHER.build_command(args, self.root),
            [sys.executable, str(self.root / "tests/run.py"), *args],
        )

    def test_valid_platform_ids_use_fallback_without_a_hook(self):
        for platform in ("cuda", "bi_v150", "future-backend", "backend2"):
            with self.subTest(platform=platform):
                args = ["--platform", platform]
                self.assertEqual(
                    DISPATCHER.build_command(args, self.root)[1],
                    str(self.root / "tests/run.py"),
                )

    def test_directory_with_runner_name_is_not_a_hook(self):
        (self.root / ".github/scripts/iluvatar/run_tests.py").mkdir(parents=True)
        self.assertEqual(
            DISPATCHER.build_command(["--platform", "iluvatar"], self.root)[1],
            str(self.root / "tests/run.py"),
        )

    def test_invalid_platform_ids_are_rejected(self):
        for platform in (
            "",
            "..",
            "../iluvatar",
            "cuda/extra",
            "cuda\\extra",
            "C:cuda",
            "cuda\n",
            "-cuda",
        ):
            with (
                self.subTest(platform=platform),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    DISPATCHER.build_command([f"--platform={platform}"], self.root)
                self.assertEqual(raised.exception.code, 2)

    def test_platform_is_required_and_cannot_be_abbreviated(self):
        for args in ([], ["--plat", "cuda"], ["--", "--platform", "cuda"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    DISPATCHER.build_command(args, self.root)
                self.assertEqual(raised.exception.code, 2)

    def test_main_replaces_process_using_current_python(self):
        args = ["--platform", "cuda", "--scope", "functional"]
        with (
            mock.patch.object(DISPATCHER.sys, "platform", "linux"),
            mock.patch.object(DISPATCHER.os, "execv") as execute,
        ):
            DISPATCHER.main(args)
        self.assertEqual(execute.call_args.args[0], sys.executable)
        self.assertEqual(execute.call_args.args[1][2:], args)

    def test_windows_process_wait_preserves_failure_status(self):
        args = ["--platform", "cuda", "--scope", "unit"]
        with (
            mock.patch.object(DISPATCHER.sys, "platform", "win32"),
            mock.patch.object(DISPATCHER.subprocess, "call", return_value=9) as execute,
            self.assertRaises(SystemExit) as raised,
        ):
            DISPATCHER.main(args)
        self.assertEqual(raised.exception.code, 9)
        self.assertEqual(execute.call_args.args[0][2:], args)

    def test_real_fallback_preserves_arguments_and_failure_status(self):
        self.write_runner("tests/run.py", "generic", 7)
        args = ["--platform", "cuda", "--scope", "unit", "--", "-k", "real case"]
        result = self.run_dispatcher(args)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"target": "generic", "argv": args})

    def test_real_hook_preserves_arguments_and_failure_status(self):
        self.write_runner("tests/run.py", "generic", 7)
        self.write_runner(".github/scripts/iluvatar/run_tests.py", "iluvatar", 5)
        args = [
            "--platform=iluvatar",
            "--scope",
            "benchmark",
            "--",
            "--junitxml=a b.xml",
        ]
        result = self.run_dispatcher(args)
        self.assertEqual(result.returncode, 5, result.stderr)
        self.assertEqual(
            json.loads(result.stdout), {"target": "iluvatar", "argv": args}
        )

    def test_invalid_platform_never_executes_a_runner(self):
        self.write_runner("tests/run.py", "generic", 0)
        result = self.run_dispatcher(["--platform=../cuda"])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
