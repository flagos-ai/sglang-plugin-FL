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
        [sys.executable, "-c", "import sglang_fl; print(sglang_fl.SOURCE)"],
        capture_output=True, text=True,
    )
    print(parent + "|" + child.stdout.strip())
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


def _run(script: str, *, cwd: Path, env_extra: dict[str, str], arg: str | None = None
         ) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_extra}
    env.pop("PYTHONSAFEPATH", None)
    cmd = [sys.executable, "-c", script]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


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
        env_extra={"PYTHONPATH": f"{staging}{os.pathsep}{isolated}", "PYTHONSAFEPATH": "1"},
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
        env_extra={"PYTHONPATH": f"{staging}{os.pathsep}{isolated}", "PYTHONSAFEPATH": "1"},
    )
    assert result.returncode == 0, result.stderr
    parent, helper, grandchild = result.stdout.strip().split("|")
    assert parent == WHEEL_SENTINEL
    assert helper == "helper"
    assert grandchild == WHEEL_SENTINEL


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
        env_extra={"PYTHONPATH": f"{staging}{os.pathsep}{checkout}", "PYTHONSAFEPATH": "1"},
        arg=str(checkout),
    )
    assert result.returncode != 0, (
        "guard silently accepted a checkout resolution:\n" + result.stdout
    )
    assert "FORBIDDEN" in result.stderr
