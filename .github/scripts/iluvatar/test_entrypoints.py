# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Bash entrypoints without installing packages or using GPUs.

Linux CI requires Bash. Windows developers can select Git Bash with
ILUVATAR_TEST_BASH; ILUVATAR_TEST_TMPDIR optionally relocates temporary files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


_ROOT = Path(__file__).resolve().parents[3]
_ORIGINAL_PYTHONPATH = "/vendor/corex/python:/vendor/flagtree/python"
_ORIGINAL_PER_OP = "rms_norm=flagos"
_INVENTORY = (
    "index, memory.used [MiB], memory.total [MiB]\n"
    "0, 24000 MiB, 32768 MiB\n"
    + "".join(f"{index}, 0 MiB, 32768 MiB\n" for index in (3, 4, 5, 6, 7))
)

# This is the only Python interpreter reached by the shell entrypoints. Every
# accepted command is handled with stdlib fakes; unknown commands fail closed.
_DISPATCH = r"""
import builtins
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def record(kind, **extra):
    event = {
        "kind": kind,
        "args": sys.argv[1:],
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "per_op": os.environ.get("SGLANG_FL_PER_OP", ""),
        **extra,
    }
    with open(os.environ["ILUVATAR_FAKE_CALLS"], "a", encoding="utf8") as output:
        output.write(json.dumps(event) + "\n")


args = sys.argv[1:]
if args == ["scripts/iluvatar/require_idle_gpus.py"]:
    record("idle")

    def inventory(command, **kwargs):
        assert command == [
            "ixsmi", "--query-gpu=index,memory.used,memory.total", "--format=csv"
        ], command
        assert kwargs["check"] and kwargs["capture_output"] and kwargs["text"]
        record("inventory")
        if os.environ.get("ILUVATAR_FAKE_INVENTORY_FAIL"):
            raise subprocess.CalledProcessError(7, command)
        return subprocess.CompletedProcess(
            command, 0, stdout=os.environ["ILUVATAR_FAKE_INVENTORY"], stderr=""
        )

    with patch.object(subprocess, "run", side_effect=inventory):
        runpy.run_path(args[0], run_name="__main__")
elif args[:3] == ["-m", "pip", "install"]:
    record("install")
    raise SystemExit(int(os.environ.get("ILUVATAR_FAKE_INSTALL_RC", "0")))
elif args and args[0] == "scripts/iluvatar/verify_environment.py":
    profile = args[args.index("--fused-profile") + 1]
    record("verify", profile=profile, minimum=args[args.index("--min-devices") + 1])
    print(json.dumps({"fake_profile": profile}))
    raise SystemExit(23 if os.environ.get("ILUVATAR_FAKE_PROFILE_FAIL") == profile else 0)
elif args == ["-"]:
    record("inline")
    source = sys.stdin.read()

    class Tensor:
        def __init__(self, values):
            self.values = values

        def square(self):
            return Tensor([value * value for value in self.values])

        def sum(self):
            return Tensor([sum(self.values)])

        def item(self):
            return self.values[0] + int(os.environ.get("ILUVATAR_FAKE_ARITHMETIC_ERROR", "0"))

    count = int(os.environ.get(
        "ILUVATAR_FAKE_GPU_COUNT", str(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
    ))
    torch = ModuleType("torch")
    torch.__version__ = "fake-corex"
    torch.float32 = "float32"
    if os.environ.get("ILUVATAR_FAKE_COREX", "1") == "1":
        torch.corex = object()

    def arange(n, *, device, dtype):
        logical = int(device.split(":")[1])
        assert 0 <= logical < count
        record("tensor", logical=logical)
        return Tensor(list(range(n)))

    torch.arange = arange
    torch.cuda = SimpleNamespace(
        is_available=lambda: os.environ.get("ILUVATAR_FAKE_CUDA_AVAILABLE", "1") == "1",
        device_count=lambda: count,
        synchronize=lambda device: None,
        get_device_name=lambda device: f"fake-{device}",
    )
    modules = {"torch": torch}
    root = Path.cwd().resolve()
    for name in ("sglang_fl", "_iluvatar_empty_compat", "sgl_kernel", "flashinfer"):
        module = ModuleType(name)
        directory = root / (
            "sglang_fl" if name == "sglang_fl" else f"addon/iluvatar-compat/src/{name}"
        )
        if os.environ.get("ILUVATAR_FAKE_STALE_MODULE") == name:
            directory = root.parent / "old-image" / name
        module.__file__ = str(directory / "__init__.py")
        module.__flagos_import_shim__ = True
        modules[name] = module

    original_import = builtins.__import__

    def track_import(name, *arguments, **kwargs):
        if name == "torch":
            record("torch-import")
        return original_import(name, *arguments, **kwargs)

    with (
        patch.dict(sys.modules, modules),
        patch.object(builtins, "__import__", side_effect=track_import),
        patch.object(importlib.metadata, "version", return_value="fake-pinned"),
    ):
        exec(compile(source, "<entrypoint-inline>", "exec"))
else:
    raise AssertionError(f"Unexpected Python command: {args}")
"""


def _shell_path(path: Path) -> str:
    """Use POSIX paths for Bash, including Git Bash on Windows."""
    value = path.resolve().as_posix()
    if os.name == "nt" and len(path.drive) == 2 and path.drive.endswith(":"):
        return f"/{value[0].lower()}{value[2:]}"
    return value


class IluvatarEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("ILUVATAR_TEST_BASH", "bash")
        bash = shutil.which(requested)
        if bash is None:
            raise RuntimeError(
                "Bash is required for entrypoint tests; Windows developers can "
                "set ILUVATAR_TEST_BASH to the Git Bash executable"
            )
        if os.name == "nt" and Path(bash).parent.name.lower() == "system32":
            raise RuntimeError(
                "Set ILUVATAR_TEST_BASH to Git Bash, not the WSL launcher"
            )
        cls.bash = bash

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="iluvatar-entrypoints-", dir=os.environ.get("ILUVATAR_TEST_TMPDIR")
        )
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "checkout"
        self.scripts = self.repo / ".github/scripts/iluvatar"
        self.scripts.mkdir(parents=True)
        for name in ("check.sh", "setup.sh"):
            shutil.copyfile(
                _ROOT / ".github/scripts/iluvatar" / name, self.scripts / name
            )
        guard = self.repo / "scripts/iluvatar/require_idle_gpus.py"
        guard.parent.mkdir(parents=True)
        shutil.copyfile(_ROOT / "scripts/iluvatar/require_idle_gpus.py", guard)
        dispatch = self.base / "dispatch.py"
        dispatch.write_text(textwrap.dedent(_DISPATCH), encoding="utf8")
        binaries = self.base / "fake-bin"
        binaries.mkdir()
        for name in ("python", "python3"):
            executable = binaries / name
            executable.write_text(
                "#!/usr/bin/env bash\n"
                'exec "$ILUVATAR_FAKE_PYTHON" -S "$ILUVATAR_FAKE_DISPATCH" "$@"\n',
                encoding="utf8",
            )
            executable.chmod(0o755)
        for name in ("pip", "pip3", "ixsmi"):
            executable = binaries / name
            executable.write_text(
                '#!/usr/bin/env bash\necho "Unexpected real-tool invocation" >&2\nexit 97\n',
                encoding="utf8",
            )
            executable.chmod(0o755)
        self.calls = self.base / "calls.jsonl"
        self.github_env = self.base / "github.env"
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ILUVATAR_FAKE_", "SGLANG_ILUVATAR_"))
        }
        self.env.update(
            PATH=f"{_shell_path(binaries)}:/usr/bin:/bin",
            MSYS2_ENV_CONV_EXCL="PYTHONPATH",
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="3,4,5,6",
            PYTHONPATH=_ORIGINAL_PYTHONPATH,
            SGLANG_FL_PER_OP=_ORIGINAL_PER_OP,
            GITHUB_ENV=_shell_path(self.github_env),
            ILUVATAR_FAKE_PYTHON=Path(sys.executable).as_posix(),
            ILUVATAR_FAKE_DISPATCH=str(dispatch),
            ILUVATAR_FAKE_CALLS=str(self.calls),
            ILUVATAR_FAKE_INVENTORY=_INVENTORY,
        )

    def run_entrypoint(self, script, *, updates=None, expected=0):
        env = {**self.env, **(updates or {})}
        result = subprocess.run(
            [self.bash, "--noprofile", "--norc", _shell_path(self.scripts / script)],
            cwd=self.base,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        log = self.repo / "artifacts/iluvatar" / Path(script).with_suffix(".log")
        self.assertTrue(log.is_file(), result.stdout + result.stderr)
        self.assertGreater(log.stat().st_size, 0)
        return (
            [
                json.loads(line)
                for line in self.calls.read_text(encoding="utf8").splitlines()
            ]
            if self.calls.exists()
            else []
        )

    def assert_no_persisted_allocation(self, events):
        self.assertFalse(self.github_env.exists())
        self.assertNotIn("tensor", [event["kind"] for event in events])

    def test_check_default_four_devices_and_persisted_allocation(self):
        events = self.run_entrypoint("check.sh")
        self.assertEqual(
            [event["logical"] for event in events if event["kind"] == "tensor"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            self.github_env.read_text(encoding="utf8"),
            "CUDA_VISIBLE_DEVICES=3,4,5,6\nSGLANG_ILUVATAR_MIN_DEVICES=4\n",
        )

    def test_check_single_operator_device(self):
        events = self.run_entrypoint(
            "check.sh",
            updates={"CUDA_VISIBLE_DEVICES": "7", "SGLANG_ILUVATAR_MIN_DEVICES": "1"},
        )
        self.assertEqual(
            [event["logical"] for event in events if event["kind"] == "tensor"], [0]
        )
        self.assertEqual(
            self.github_env.read_text(encoding="utf8"),
            "CUDA_VISIBLE_DEVICES=7\nSGLANG_ILUVATAR_MIN_DEVICES=1\n",
        )

    def test_check_missing_allocation_fails_before_python(self):
        events = self.run_entrypoint(
            "check.sh", updates={"CUDA_VISIBLE_DEVICES": ""}, expected=2
        )
        self.assertEqual(events, [])
        self.assert_no_persisted_allocation(events)

    def test_check_implicit_allocation_fails_before_python(self):
        events = self.run_entrypoint(
            "check.sh", updates={"CUDA_VISIBLE_DEVICES": "all"}, expected=2
        )
        self.assertEqual(events, [])
        self.assert_no_persisted_allocation(events)

    def test_check_busy_allocation_fails_before_torch_import(self):
        events = self.run_entrypoint(
            "check.sh", updates={"CUDA_VISIBLE_DEVICES": "0"}, expected=1
        )
        self.assertEqual([event["kind"] for event in events], ["idle", "inventory"])
        self.assert_no_persisted_allocation(events)

    def test_check_duplicate_allocation_fails_before_torch_import(self):
        events = self.run_entrypoint(
            "check.sh", updates={"CUDA_VISIBLE_DEVICES": "3,3"}, expected=1
        )
        self.assertEqual([event["kind"] for event in events], ["idle", "inventory"])
        self.assert_no_persisted_allocation(events)

    def test_check_inventory_failure_fails_before_torch_import(self):
        events = self.run_entrypoint(
            "check.sh", updates={"ILUVATAR_FAKE_INVENTORY_FAIL": "1"}, expected=1
        )
        self.assertEqual([event["kind"] for event in events], ["idle", "inventory"])
        self.assert_no_persisted_allocation(events)

    def test_check_invalid_minimum_fails_before_python(self):
        events = self.run_entrypoint(
            "check.sh", updates={"SGLANG_ILUVATAR_MIN_DEVICES": "0"}, expected=2
        )
        self.assertEqual(events, [])
        self.assert_no_persisted_allocation(events)

    def test_check_too_few_devices_fails_before_tensor_operations(self):
        events = self.run_entrypoint(
            "check.sh", updates={"CUDA_VISIBLE_DEVICES": "7"}, expected=1
        )
        self.assertIn("torch-import", [event["kind"] for event in events])
        self.assert_no_persisted_allocation(events)

    def test_check_stock_torch_is_rejected(self):
        events = self.run_entrypoint(
            "check.sh", updates={"ILUVATAR_FAKE_COREX": "0"}, expected=1
        )
        self.assertIn("torch-import", [event["kind"] for event in events])
        self.assert_no_persisted_allocation(events)

    def test_check_unavailable_cuda_alias_is_rejected(self):
        events = self.run_entrypoint(
            "check.sh", updates={"ILUVATAR_FAKE_CUDA_AVAILABLE": "0"}, expected=1
        )
        self.assertIn("torch-import", [event["kind"] for event in events])
        self.assert_no_persisted_allocation(events)

    def test_check_unexpected_visibility_is_rejected_before_tensor_operations(self):
        events = self.run_entrypoint(
            "check.sh", updates={"ILUVATAR_FAKE_GPU_COUNT": "8"}, expected=1
        )
        self.assertIn("torch-import", [event["kind"] for event in events])
        self.assert_no_persisted_allocation(events)

    def test_check_bad_arithmetic_propagates_failure(self):
        events = self.run_entrypoint(
            "check.sh", updates={"ILUVATAR_FAKE_ARITHMETIC_ERROR": "1"}, expected=1
        )
        self.assertEqual(
            [event["logical"] for event in events if event["kind"] == "tensor"], [0]
        )
        self.assertFalse(self.github_env.exists())

    def test_setup_preserves_dependencies_paths_and_isolates_flagos_override(self):
        events = self.run_entrypoint("setup.sh")
        self.assertEqual(
            [event["kind"] for event in events],
            ["install", "inline", "verify", "verify"],
        )
        install = events[0]["args"]
        self.assertTrue(
            {"--no-index", "--no-deps", "--no-build-isolation", "--force-reinstall"}
            <= set(install)
        )
        self.assertEqual(
            [install[index + 1] for index, arg in enumerate(install) if arg == "-e"],
            [".", "./addon/iluvatar-compat"],
        )
        profiles = [event for event in events if event["kind"] == "verify"]
        self.assertEqual([event["profile"] for event in profiles], ["vendor", "flagos"])
        self.assertEqual([event["minimum"] for event in profiles], ["4", "4"])
        self.assertEqual(profiles[0]["per_op"], _ORIGINAL_PER_OP)
        self.assertEqual(
            dict(pair.split("=", 1) for pair in profiles[1]["per_op"].split(";")),
            {
                "gemma_rms_norm": "reference",
                "fused_moe": "flagos",
                "chunk_gated_delta_rule": "flagos",
                "fused_recurrent_gated_delta_rule": "flagos",
                "fused_recurrent_gated_delta_rule_packed_decode": "flagos",
            },
        )
        expected_path = (
            f"{_shell_path(self.repo)}:"
            f"{_shell_path(self.repo / 'addon/iluvatar-compat/src')}:"
            f"{_ORIGINAL_PYTHONPATH}"
        )
        self.assertEqual(
            [event["pythonpath"] for event in profiles], [expected_path] * 2
        )
        self.assertEqual(
            self.github_env.read_text(encoding="utf8"), f"PYTHONPATH={expected_path}\n"
        )
        for profile in ("vendor", "flagos"):
            log = self.repo / f"artifacts/iluvatar/{profile}-environment.log"
            self.assertTrue(log.is_file())
            self.assertEqual(json.loads(log.read_text()), {"fake_profile": profile})

    def test_setup_single_operator_device_reaches_both_profiles(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"SGLANG_ILUVATAR_MIN_DEVICES": "1"}
        )
        self.assertEqual(
            [event["minimum"] for event in events if event["kind"] == "verify"],
            ["1", "1"],
        )

    def test_setup_install_failure_stops_all_preflights(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"ILUVATAR_FAKE_INSTALL_RC": "19"}, expected=19
        )
        self.assertEqual([event["kind"] for event in events], ["install"])
        self.assertFalse(self.github_env.exists())

    def test_setup_stale_plugin_is_rejected_before_profiles(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"ILUVATAR_FAKE_STALE_MODULE": "sglang_fl"}, expected=1
        )
        self.assertEqual([event["kind"] for event in events], ["install", "inline"])

    def test_setup_stale_addon_is_rejected_before_profiles(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"ILUVATAR_FAKE_STALE_MODULE": "sgl_kernel"}, expected=1
        )
        self.assertEqual([event["kind"] for event in events], ["install", "inline"])

    def test_setup_vendor_failure_propagates_without_flagos_preflight(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"ILUVATAR_FAKE_PROFILE_FAIL": "vendor"}, expected=23
        )
        self.assertEqual(
            [event["kind"] for event in events], ["install", "inline", "verify"]
        )
        self.assertEqual(events[-1]["profile"], "vendor")
        self.assertFalse(
            (self.repo / "artifacts/iluvatar/flagos-environment.log").exists()
        )

    def test_setup_flagos_failure_propagates_through_tee(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"ILUVATAR_FAKE_PROFILE_FAIL": "flagos"}, expected=23
        )
        self.assertEqual(
            [event["kind"] for event in events],
            ["install", "inline", "verify", "verify"],
        )
        self.assertEqual(events[-1]["profile"], "flagos")

    def test_setup_invalid_minimum_fails_before_installation(self):
        events = self.run_entrypoint(
            "setup.sh", updates={"SGLANG_ILUVATAR_MIN_DEVICES": "0"}, expected=2
        )
        self.assertEqual(events, [])
        self.assertFalse(self.github_env.exists())


if __name__ == "__main__":
    unittest.main()
