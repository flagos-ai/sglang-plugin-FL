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
* child and grandchild safe-path states are read from the real
  ``sys.flags.safe_path``; interpreters without the flag (Python < 3.11)
  only get a clearly-labeled PYTHONSAFEPATH propagation check;
* if the checkout copy still wins in installed-wheel mode, the guard
  script must exit non-zero instead of silently continuing.

The fake install location stands in for site-packages on PYTHONPATH; the
ordering property under test (nothing from the checkout is on the child
path) is the same one that makes the real wheel win in CI.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import pytest

CHECKOUT_SENTINEL = "checkout"
WHEEL_SENTINEL = "wheel"


def classify_safe_path(safe_path_flag, env_value):
    """Classify safe-path from the real interpreter flag.

    ``sys.flags.safe_path`` (Python >= 3.11) is authoritative: ``True``
    passes, ``False`` fails. Inferring from cwd versus ``sys.path[0]``
    misjudges safe-path as disabled when PYTHONPATH legitimately puts the
    staging dir (== cwd) first -- the operations Runtime's 3.12 probe showed
    ``safe_path=True`` with ``cwd == sys.path[0]``. Older interpreters
    have no flag to read; there we only verify that ``PYTHONSAFEPATH``
    propagated and label the report as compatibility-only, never
    claiming safe-path is actually enabled.

    Returns ``(state, ok)``.
    """
    if safe_path_flag is not None:
        # The flag is authoritative regardless of the env; bool() keeps
        # this correct for int-flagged builds too (0/1 instead of
        # True/False).
        if safe_path_flag:
            return ("flag=true", True)
        return ("flag=false", False)
    if env_value == "1":
        return ("flag unsupported; env propagated", True)
    return ("flag unsupported; env missing", False)


def expected_child_safe_path(env_value, child_version=sys.version_info):
    """Expected safe-path state of a child STARTED with this env.

    The child runs this test's sys.executable, so the interpreter
    VERSION (stable across processes) decides whether the child has
    ``sys.flags.safe_path`` -- but the parent's own flag VALUE must
    never be reused: it reflects the parent's launch conditions. The
    operations Runtime 3.12 probe measured a pytest parent at
    ``safe_path=False`` / ``PYTHONSAFEPATH=None`` while a child started
    with ``PYTHONSAFEPATH=1`` reads ``safe_path=True``; deriving the
    expectation from the parent's flag would demand ``flag=false``
    from a child that correctly reports ``flag=true``.

    Predicts from the child's launch conditions only:
      * >= 3.11: env "1" -> the child's real flag is True; otherwise
        False (the env var is what switches the flag on);
      * <  3.11: no flag to read; labeled env fallback.

    classify_safe_path remains the authority for OBSERVED states -- a
    real flag of False is rejected even when the env text claims "1".
    """
    if child_version >= (3, 11):
        if env_value == "1":
            return ("flag=true", True)
        return ("flag=false", False)
    if env_value == "1":
        return ("flag unsupported; env propagated", True)
    return ("flag unsupported; env missing", False)


# Standalone copies of classify_safe_path for the subprocess probes below
# (they run under sys.executable with the wheel-mode env and cannot import
# this test module). TestSafePathProbeAgreement pins them to the classifier.
_SAFE_PATH_PROBE = """
def _safe_path_state():
    flag = getattr(sys.flags, "safe_path", None)
    if flag is not None:
        if flag:
            return ("flag=true", True)
        return ("flag=false", False)
    if os.environ.get("PYTHONSAFEPATH") == "1":
        return ("flag unsupported; env propagated", True)
    return ("flag unsupported; env missing", False)
"""

# The grandchild runs as a bare `python -c` one-liner: the same mapping
# compressed into a conditional expression. The import header is separate
# so the agreement test can execute just the classifier against fakes.
_GRANDCHILD_CLASSIFIER = (
    "flag = getattr(sys.flags, 'safe_path', None); "
    "env = os.environ.get('PYTHONSAFEPATH'); "
    "print(sglang_fl.SOURCE, "
    "('flag=true' if flag else 'flag=false') if flag is not None else "
    "('flag unsupported; env propagated' if env == '1' "
    "else 'flag unsupported; env missing'))"
)
_GRANDCHILD_CMD = "import os, sglang_fl, sys; " + _GRANDCHILD_CLASSIFIER

GUARD_SCRIPT = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    + _SAFE_PATH_PROBE
    + textwrap.dedent(
        """
        state, ok = _safe_path_state()
        if not ok:
            print(f"wheel mode requires safe-path, got: {state}", file=sys.stderr)
            sys.exit(2)
        print(
            f"safe-path: {state} "
            f"(PYTHONSAFEPATH={os.environ.get('PYTHONSAFEPATH')})",
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
)

_CHILD_BODY = textwrap.dedent(
    """
    import sglang_fl, tests.helper
    parent = f"{sglang_fl.SOURCE}|{tests.helper.LABEL}"
    state, _ok = _safe_path_state()
    grandchild = subprocess.run(
        [sys.executable, "-c", GRANDCHILD_CMD],
        capture_output=True, text=True,
    )
    print(parent + "|" + state + "|" + grandchild.stdout.strip())
    """
)

CHILD_SCRIPT = (
    "import os\n"
    "import subprocess\n"
    "import sys\n"
    + _SAFE_PATH_PROBE
    + f"GRANDCHILD_CMD = {_GRANDCHILD_CMD!r}\n"
    + _CHILD_BODY
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


class TestAbsolutizeIgnoreFlags:
    def test_relative_filters_reanchored_to_repo_root(self):
        run = _load_run_py()
        cmd = [
            "python",
            "-m",
            "pytest",
            str(run._REPO_ROOT / "tests" / "unit_tests"),
            "--ignore",
            "tests/unit_tests/kunlunxin",
            "--ignore-glob",
            "tests/functional_tests/ops/test_*.py",
            "-q",
        ]
        out = run._absolutize_ignore_flags(list(cmd))
        assert out[5] == str((run._REPO_ROOT / "tests/unit_tests/kunlunxin").resolve())
        assert out[7] == str(
            (run._REPO_ROOT / "tests/functional_tests/ops/test_*.py").resolve()
        )
        # everything except the filter values is untouched
        assert [v for i, v in enumerate(out) if i not in (5, 7)] == [
            v for i, v in enumerate(cmd) if i not in (5, 7)
        ]

    def test_absolute_filters_and_other_flags_pass_through(self):
        run = _load_run_py()
        absolute = "/somewhere/else"
        cmd = ["python", "-m", "pytest", "x", "--ignore", absolute, "--tb=short"]
        assert run._absolutize_ignore_flags(list(cmd)) == cmd

    def test_trailing_ignore_without_value_is_left_alone(self):
        run = _load_run_py()
        cmd = ["python", "-m", "pytest", "x", "--ignore"]
        assert run._absolutize_ignore_flags(list(cmd)) == cmd


class TestGuardSeamIsolation:
    def test_guard_probe_uses_the_real_subprocess_run(self, monkeypatch):
        # In-process drivers of main() intercept subprocess.run to observe
        # exactly the case launches (test_iluvatar_benchmark_config pins
        # one launch per case). The guard's pre-flight probe is an
        # internal diagnostic and must go through the reference bound at
        # import time, never through the interceptable module attribute.
        run = _load_run_py()
        recorded = []

        def real_run(command, **kwargs):
            recorded.append((command, kwargs))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        def unexpected(*args, **kwargs):
            raise AssertionError(
                "the guard probe must not ride the intercepted "
                "subprocess.run seam reserved for case launches"
            )

        monkeypatch.setattr(run.subprocess, "run", unexpected)
        monkeypatch.setattr(run, "_REAL_SUBPROCESS_RUN", real_run)
        run._guard_installed_plugin({"PATH": "/usr/bin"})

        assert len(recorded) == 1
        command, kwargs = recorded[0]
        assert command == [
            run.sys.executable,
            "-c",
            run._GUARD_SCRIPT,
            str(run._REPO_ROOT),
        ]
        assert kwargs["cwd"] == str(run._wheel_staging_dir())
        assert kwargs["env"] == {"PATH": "/usr/bin"}
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True


class TestClassifySafePath:
    def test_flag_true_passes(self):
        # Python 3.12 with PYTHONSAFEPATH (or -P): the real flag is True.
        assert classify_safe_path(True, "1") == ("flag=true", True)
        # `python -P` enables the flag without the env var.
        assert classify_safe_path(True, None) == ("flag=true", True)

    def test_flag_false_fails(self):
        # Python 3.12 without safe-path: the real flag is False and the
        # check must fail regardless of what the env claims.
        assert classify_safe_path(False, None) == ("flag=false", False)
        assert classify_safe_path(False, "1") == ("flag=false", False)

    def test_old_python_env_propagated_is_compatibility_only(self):
        state, ok = classify_safe_path(None, "1")
        assert ok
        # The report must not claim safe-path is enabled on an
        # interpreter that has no flag to read.
        assert state == "flag unsupported; env propagated"

    def test_old_python_env_missing_fails(self):
        assert classify_safe_path(None, None) == (
            "flag unsupported; env missing",
            False,
        )
        assert classify_safe_path(None, "") == ("flag unsupported; env missing", False)


class TestExpectedChildSafePath:
    """Child expectations derive from the child's launch conditions.

    These are pinned for BOTH interpreter generations via the
    child_version parameter so a 3.9 host still exercises the 3.11+
    expectations the operations Runtime CI will actually run.
    """

    def test_flag_interpreter_with_env_expects_true(self):
        # Parent/child share only the interpreter version; the child,
        # started with PYTHONSAFEPATH=1, must read the real flag as True
        # even when the (envless) parent reads False.
        assert expected_child_safe_path("1", (3, 12, 13)) == ("flag=true", True)
        # The prediction must match what the child's own classifier
        # observes once the env has switched its flag on.
        assert classify_safe_path(True, "1") == expected_child_safe_path(
            "1", (3, 12, 13)
        )

    def test_flag_interpreter_without_env_expects_false(self):
        assert expected_child_safe_path(None, (3, 12, 13)) == ("flag=false", False)
        assert expected_child_safe_path("", (3, 12, 13)) == ("flag=false", False)

    def test_old_interpreter_env_propagated(self):
        assert expected_child_safe_path("1", (3, 9, 6)) == (
            "flag unsupported; env propagated",
            True,
        )

    def test_old_interpreter_env_missing(self):
        assert expected_child_safe_path(None, (3, 9, 6)) == (
            "flag unsupported; env missing",
            False,
        )


class TestSafePathProbeAgreement:
    """The standalone subprocess snippets must mirror classify_safe_path."""

    @pytest.mark.parametrize(
        ("flag", "env"),
        [
            (True, "1"),
            (True, None),
            (False, None),
            (False, "1"),
            ("missing", "1"),
            ("missing", None),
            ("missing", ""),
        ],
    )
    def test_probe_and_grandchild_match_classifier(self, flag, env):
        fake_os = types.SimpleNamespace(
            environ={} if env is None else {"PYTHONSAFEPATH": env}
        )
        flags = (
            types.SimpleNamespace()
            if flag == "missing"
            else (types.SimpleNamespace(safe_path=flag))
        )
        fake_sys = types.SimpleNamespace(flags=flags)
        real_flag = None if flag == "missing" else flag

        namespace: dict = {"os": fake_os, "sys": fake_sys}
        exec(_SAFE_PATH_PROBE, namespace)  # noqa: S102 - our own constant
        assert namespace["_safe_path_state"]() == classify_safe_path(real_flag, env)

        # The classifier tail of the grandchild one-liner (without its
        # import header, which would rebind these fakes to the real
        # modules) prints "<SOURCE> <state>"; print() resolves sys.stdout
        # on the real module, so redirect it.
        namespace = {
            "os": fake_os,
            "sys": fake_sys,
            "sglang_fl": types.SimpleNamespace(SOURCE=WHEEL_SENTINEL),
        }
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            exec(_GRANDCHILD_CLASSIFIER, namespace)  # noqa: S102 - own constant
        assert captured.getvalue().strip() == (
            f"{WHEEL_SENTINEL} {classify_safe_path(real_flag, env)[0]}"
        )


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
    # The guard reports the real interpreter flag (or, on interpreters
    # without one, the labeled env-propagation fallback) -- never a
    # cwd/sys.path[0] guess. The expectation comes from the CHILD's
    # launch conditions, not this process's own flags.
    expected_state, ok = expected_child_safe_path("1")
    assert ok
    assert f"safe-path: {expected_state} (PYTHONSAFEPATH=1)" in result.stderr


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
    # The "|" delimiters are unambiguous even though the state tokens
    # themselves contain spaces ("flag unsupported; env propagated").
    parent, helper, child_state, grandchild_line = result.stdout.strip().split("|")
    assert parent == WHEEL_SENTINEL
    assert helper == "helper"
    # Both generations must report the state of THEIR OWN processes.
    # Sharing sys.executable only guarantees the same interpreter
    # VERSION -- it says nothing about the flag VALUES: this pytest
    # parent typically runs without PYTHONSAFEPATH (safe_path=False on
    # 3.12) while the child, started with the env, reads safe_path=True
    # (measured on the operations Runtime 3.12). Expectations therefore
    # come from the child's launch conditions via
    # expected_child_safe_path, never from this process's flags.
    expected_state, ok = expected_child_safe_path("1")
    assert ok
    assert child_state == expected_state, f"child safe-path: {child_state}"
    assert grandchild_line == f"{WHEEL_SENTINEL} {expected_state}", (
        f"grandchild safe-path: {grandchild_line}"
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


def test_wheel_mode_guard_rejects_missing_safe_path(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _write_package(checkout, CHECKOUT_SENTINEL)
    _write_tests_package(checkout)
    isolated = tmp_path / "site"
    _write_package(isolated, WHEEL_SENTINEL)

    staging = Path(tempfile.mkdtemp(prefix="wheel-mode-test."))
    (staging / "tests").symlink_to((checkout / "tests").resolve())

    # Wheel-mode env without PYTHONSAFEPATH: on >= 3.11 the real
    # sys.flags.safe_path reads False ("flag=false"); on older
    # interpreters the env is the only carrier ("flag unsupported; env
    # missing"). Either way the guard must refuse before any import.
    result = _run(
        GUARD_SCRIPT,
        cwd=staging,
        env_extra={"PYTHONPATH": f"{staging}{os.pathsep}{isolated}"},
        arg=str(checkout),
    )
    expected_state, ok = expected_child_safe_path(None)
    assert not ok
    assert result.returncode == 2, result.stderr
    assert f"wheel mode requires safe-path, got: {expected_state}" in result.stderr
    assert result.stdout == ""  # refused before importing/reporting anything
