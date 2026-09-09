"""Protect phase failures, device allocation and task-scoped cleanup."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import container


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.environment = {
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_WORKSPACE": str(root / "source"),
            "RUNNER_TEMP": str(root / "temp"),
            "GITHUB_ENV": str(root / "temp" / "environment"),
            "MUSA_CI_IMAGE_ID": "sha256:verified-config",
            "MUSA_VISIBLE_DEVICES": "2,3,6,7",
            "MUSA_CI_CONFIG": json.dumps(
                {
                    "container_options": "--init --ipc=host",
                    "container_volumes": ["/data:/data"],
                }
            ),
        }

    def test_start_uses_verified_image_and_preserves_allocation(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch.object(container.subprocess, "run") as run:
                container.start()
        args = run.call_args.args[0]
        self.assertIn("sha256:verified-config", args)
        self.assertIn("MUSA_VISIBLE_DEVICES=2,3,6,7", args)
        self.assertIn("musa-ci-123-2", args)
        self.assertIn("/data:/data", args)

    def test_unmounted_environment_file_fails_before_start(self):
        environment = {**self.environment, "GITHUB_ENV": self.temp.name + "/outside"}
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(container.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    container.start()
                run.assert_not_called()

    def test_model_failure_and_selected_environment_reach_the_job(self):
        environment = {**self.environment, "PYTHONPATH": "/workspace:/opt/FlagGems/src"}
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(container.subprocess, "call", return_value=42) as call:
                self.assertEqual(container.execute(["python", "tests/run.py"]), 42)
        args = call.call_args.args[0]
        self.assertIn("MUSA_VISIBLE_DEVICES=2,3,6,7", args)
        self.assertIn("PYTHONPATH=/workspace:/opt/FlagGems/src", args)
        self.assertEqual(args[-3:], ["musa-ci-123-2", "python", "tests/run.py"])

    def test_cleanup_only_removes_this_attempts_container(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch.object(container.subprocess, "run") as run:
                run.return_value.returncode = 1
                container.stop()
                self.assertEqual(run.call_count, 1)
                run.reset_mock()
                run.return_value.returncode = 0
                container.stop()
                self.assertEqual(
                    run.call_args.args[0],
                    ["docker", "rm", "--force", "musa-ci-123-2"],
                )


if __name__ == "__main__":
    unittest.main()
