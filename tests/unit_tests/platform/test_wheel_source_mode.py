# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel test mode: no checkout shadowing, subprocess inherited.

These are no-GPU mechanism tests. They build a fake checkout (a package
directory plus a ``tests`` package) and a fake install location holding a
different build of the same package name, then drive real subprocesses
through exactly the resolution rules tests/run.py applies:

* default mode loads the package from the checkout;
* installed-wheel mode (staging dir with a ``tests`` symlink, PYTHONPATH
  pointing at it, PYTHONSAFEPATH=1) loads the package from the isolated
  install location while ``tests.*`` helpers stay importable;
* grandchildren inherit the installed-wheel resolution;
* if the checkout copy still wins in installed-wheel mode, the guard
  script must exit non-zero instead of silently continuing.

The fake install location stands in for site-packages on PYTHONPATH; the
ordering property under test (nothing from the checkout is on the child
path) is the same one that makes the real wheel win in CI.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

CHECKOUT_SENTINEL = "checkout"
WHEEL_SENTINEL = "wheel"

GUARD_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    # Safe-path probe. PYTHONSAFEPATH lands in 3.11; on older
    # interpreters the cwd prepend cannot be disabled, so we verify the
    # env carries the flag and report the mechanism as unsupported --
    # the CI contract target is 3.12, where the strict check applies.
    import os
    supported = sys.version_info >= (3, 11)
    active = supported and bool(sys.path) and sys.path[0] != "" and (
        os.getcwd() != (sys.path[0] if sys.path else "")
    )
    env_flag = os.environ.get("PYTHONSAFEPATH") == "1"
    if supported and not active:
        print("wheel mode requires safe-path to be active", file=sys.stderr)
        sys.exit(2)
    if not supported and not env_flag:
        print("wheel mode requires PYTHONSAFEPATH=1 in the env", file=sys.stderr)
        sys.exit(2)
    print(
        "safe-path:",
        "active" if active else f"unsupported-on-{sys.version_info}",
        f"(PYTHONSAFEPATH={env_flag})",
        file=sys.stderr,
    )
    import sglang_fl
    where = Path(sglang_fl.__file__).resolve()
    checkout = Path(sys.argv[1]).resolve()
    if checkout in where.parents:
        print(f"FORBIDDEN: sglang_fl resolved from the checkout: {where}", file=sys.stderr)
        sys.exit(1)
    import tests.helper
    print(sglang_fl.SOURCE, tests.helper.LABEL, where)
    """
)

CHILD_SCRIPT = textwrap.dedent(
    """
    import subprocess, sys
    import sglang_fl, tests.helper
    parent = f"{sglang_fl.SOURCE}|{tests.helper.LABEL}"
    child = subprocess.run(
        [sys.executable, "-c",
         "import os, sglang_fl, sys; "
         "print(sglang_fl.SOURCE, "
         "'active' if (sys.version_info >= (3, 11) and sys.path "
         "and sys.path[0] not in ('', os.getcwd())) else "
         "('env' if os.environ.get('PYTHONSAFEPATH') == '1' else 'OFF'))"],
        capture_output=True, text=True,
    )
    import os
    if sys.version_info >= (3, 11):
        own = "active" if sys.path and sys.path[0] not in ("", os.getcwd()) else "OFF"
    else:
        own = "env" if os.environ.get("PYTHONSAFEPATH") == "1" else "OFF"
    print(parent + "|" + own + "|" + child.stdout.strip())
    """
)


def _write_package(root: Path, source: str) -> None:
    package = root / "sglang_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"SOURCE = {source!r}\n", encoding="utf-8")


def _write_tests_package(root: Path) -> None:
    tests = root / "tests"
    tests.mkdir(parents=True)
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "helper.py").write_text("LABEL = 'helper'\n", encoding="utf-8")


def _run(
    script: str, *, cwd: Path, env_extra: dict[str, str], arg: str | None = None
) -> subprocess.CompletedProcess:
    # Strip any inherited PYTHONSAFEPATH FIRST, then merge the explicit
    # extras, so a wheel-mode run really starts with safe-path enabled
    # (and a default-mode run really starts without it).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    env.update(env_extra)
    cmd = [sys.executable, "-c", script]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


def _load_run_py():
    spec = importlib.util.spec_from_file_location(
        "run_py_module",
        Path(__file__).resolve().parents[3] / "tests" / "run.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks the module up by name
    spec.loader.exec_module(module)
    return module


class TestFilterPythonpath:
    def test_staging_first_triton_kept_checkout_dropped(self, tmp_path):
        run = _load_run_py()
        repo = tmp_path / "repo"
        repo.mkdir()
        checkout_sub = repo / "sub"
        checkout_sub.mkdir()
        existing = os.pathsep.join(
            [
                str(tmp_path / "repo"),
                "/opt/triton",
                "",
                str(checkout_sub),
                "/some/other/lib",
            ]
        )
        kept = run._filter_pythonpath(existing, repo)
        assert kept == ["/opt/triton", "/some/other/lib"]

    def test_empty_and_whitespace_only_entries_dropped(self, tmp_path):
        run = _load_run_py()
        kept = run._filter_pythonpath(
            f" {os.pathsep} {os.pathsep}/opt/triton{os.pathsep}", tmp_path
        )
        assert kept == ["/opt/triton"]


def test_default_mode_loads_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _write_package(checkout, CHECKOUT_SENTINEL)
    _write_tests_package(checkout)
    isolated = tmp_path / "site"
    _write_package(isolated, WHEEL_SENTINEL)

    # Default: cwd=checkout, no PYTHONSAFEPATH -> the `python -c` CWD
    # prepend makes the checkout copy win, exactly like today's checkout
    # mode (and local source development).
    result = _run(
        "import sglang_fl; print(sglang_fl.SOURCE)",
        cwd=checkout,
        env_extra={"PYTHONPATH": str(isolated)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == CHECKOUT_SENTINEL


def test_wheel_mode_loads_isolated_install_and_keeps_tests_importable(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    _write_package(checkout, CHECKOUT_SENTINEL)
    _write_tests_package(checkout)
    isolated = tmp_path / "site"
    _write_package(isolated, WHEEL_SENTINEL)

    # run.py's staging dir: only a `tests` symlink, nothing else.
    staging = Path(tempfile.mkdtemp(prefix="wheel-mode-test."))
    (staging / "tests").symlink_to((checkout / "tests").resolve())

    result = _run(
        GUARD_SCRIPT,
        cwd=staging,
        env_extra={
            "PYTHONPATH": f"{staging}{os.pathsep}{isolated}",
            "PYTHONSAFEPATH": "1",
        },
        arg=str(checkout),
    )
    assert result.returncode == 0, result.stderr
    first_line = result.stdout.strip().splitlines()[0]
    sentinel, helper_label, resolved_path = first_line.split(" ", 2)
    assert sentinel == WHEEL_SENTINEL, first_line
    assert helper_label == "helper", first_line  # tests.* stayed importable
    resolved = Path(resolved_path)
    assert resolved.is_relative_to(isolated), resolved
    assert not resolved.is_relative_to(checkout), resolved


def test_wheel_mode_inherits_into_grandchildren(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _write_package(checkout, CHECKOUT_SENTINEL)
    _write_tests_package(checkout)
    isolated = tmp_path / "site"
    _write_package(isolated, WHEEL_SENTINEL)

    staging = Path(tempfile.mkdtemp(prefix="wheel-mode-test."))
    (staging / "tests").symlink_to((checkout / "tests").resolve())

    result = _run(
        CHILD_SCRIPT,
        cwd=staging,
        env_extra={
            "PYTHONPATH": f"{staging}{os.pathsep}{isolated}",
            "PYTHONSAFEPATH": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    parent, helper, parent_safe, grandchild_part = result.stdout.strip().split("|")
    grandchild, grandchild_safe = grandchild_part.split()
    assert parent == WHEEL_SENTINEL
    assert helper == "helper"
    assert parent_safe in ("active", "env"), f"child safe-path: {parent_safe}"
    assert grandchild == WHEEL_SENTINEL
    assert grandchild_safe in ("active", "env"), (
        f"grandchild safe-path: {grandchild_safe}"
    )


def test_wheel_mode_fails_loudly_when_checkout_wins(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _write_package(checkout, CHECKOUT_SENTINEL)
    _write_tests_package(checkout)

    # No isolated install at all: the staging dir keeps tests importable
    # but sglang_fl falls back to the checkout -> the guard must refuse.
    staging = Path(tempfile.mkdtemp(prefix="wheel-mode-test."))
    (staging / "tests").symlink_to((checkout / "tests").resolve())

    # PYTHONPATH deliberately includes the checkout so sglang_fl resolves
    # there -- the forbidden outcome the guard exists to catch.
    result = _run(
        GUARD_SCRIPT,
        cwd=staging,
        env_extra={
            "PYTHONPATH": f"{staging}{os.pathsep}{checkout}",
            "PYTHONSAFEPATH": "1",
        },
        arg=str(checkout),
    )
    assert result.returncode != 0, (
        "guard silently accepted a checkout resolution:\n" + result.stdout
    )
    assert "FORBIDDEN" in result.stderr
