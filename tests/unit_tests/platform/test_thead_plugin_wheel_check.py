# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the T-Head plugin-wheel acceptance check.

Builds synthetic wheels in memory and pins the contract the acceptance
review requires, most importantly that entry points are compared with the
EXACT ``(name, value)`` pair — a name-only match lets a value typo pass
and silently break plugin activation.
"""

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "thead"
    / "scripts"
    / "check_plugin_wheel.py"
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_check():
    spec = importlib.util.spec_from_file_location("check_plugin_wheel", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return _load_check()


VALID_EXTRA = [
    "sglang==0.5.18",
    "pyarrow<26",
]

VALID_ENTRY_POINTS = """\
[sglang.srt.platforms]
sglang_fl = sglang_fl:activate_platform

[sglang.srt.plugins]
sglang_fl = sglang_fl:load_plugin
"""

VALID_PAYLOAD = [
    "sglang_fl/__init__.py",
    "sglang_fl/dispatch/backends/vendor/thead/register_platform.py",
    "sglang_fl/dispatch/backends/vendor/thead/patch.py",
    "sglang_fl/dispatch/backends/vendor/thead/import_guard.py",
    "sglang_fl/dispatch/config/thead.yaml",
]


def _make_wheel(
    *,
    name="sglang_fl",
    version="0.1.0",
    extra=VALID_EXTRA,
    entry_points=VALID_ENTRY_POINTS,
    payload=VALID_PAYLOAD,
    with_tests=False,
):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        metadata_lines = [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            "Requires-Dist: flag_gems",
            "Requires-Dist: pyyaml",
        ]
        metadata_lines += [f'Requires-Dist: {req}; extra == "thead"' for req in extra]
        zf.writestr(
            "sglang_fl-0.1.0.dist-info/METADATA",
            "\n".join(metadata_lines) + "\n",
        )
        zf.writestr("sglang_fl-0.1.0.dist-info/entry_points.txt", entry_points)
        zf.writestr("sglang_fl-0.1.0.dist-info/RECORD", "")
        zf.writestr("sglang_fl-0.1.0.dist-info/WHEEL", "")
        for p in payload:
            zf.writestr(p, "#\n")
        if with_tests:
            zf.writestr("tests/unit_tests/platform/test_dummy.py", "#\n")
    return buf.getvalue()


def _run(check, wheel_bytes):
    """Run the check's main() against an in-memory wheel; return its exit code."""
    import tempfile as _t
    import sys

    fd, path = _t.mkstemp(suffix=".whl")
    try:
        with open(fd, "wb") as fh:
            fh.write(wheel_bytes)
        argv = sys.argv
        sys.argv = [
            "check_plugin_wheel.py",
            str(path),
            "--repo-root",
            str(_REPO_ROOT),
        ]
        try:
            return check.main()
        finally:
            sys.argv = argv
    finally:
        Path(path).unlink(missing_ok=True)


def test_valid_wheel_passes(check):
    assert _run(check, _make_wheel()) == 0


def test_entry_point_value_typo_fails_loudly(check):
    # Name-only acceptance would let this through; the exact (name, value)
    # contract must reject it.
    typo = VALID_ENTRY_POINTS.replace(
        "sglang_fl:activate_platform", "sglang_fl.activate_platform"
    )
    assert _run(check, _make_wheel(entry_points=typo)) == 1


def test_missing_entry_point_group_fails(check):
    partial = VALID_ENTRY_POINTS.split("[sglang.srt.plugins]")[0]
    assert _run(check, _make_wheel(entry_points=partial)) == 1


def test_missing_thead_anchor_fails(check):
    assert _run(check, _make_wheel(extra=[])) == 1


def test_thead_extra_drift_fails(check):
    drifted = VALID_EXTRA + ["datasets"]
    assert _run(check, _make_wheel(extra=drifted)) == 1


@pytest.mark.parametrize(
    "dependency",
    ["numpy==1.26.4", "triton>=3.0", "datasets==5.2.0"],
)
def test_not_plugin_dependency_rejected(check, dependency):
    # numpy is Runtime-owned and the Runtime provides the Triton compiler as
    # a module tree; datasets comes with the SGLang closure and only pyarrow
    # needs a bound on a NumPy 1.x Runtime: the plugin wheel declares none
    # of these.
    assert _run(check, _make_wheel(extra=VALID_EXTRA + [dependency])) == 1


def test_normalize_requirement_folds_clause_order(check):
    # The acceptance compares normalized requirement SETS: setuptools may
    # render `pyarrow>=24,<26` from pyproject as `pyarrow<26,>=24` in
    # METADATA; the comparison must fold clause order (and strip markers).
    assert check.normalize_requirement(
        'pyarrow>=24,<26; extra == "thead"'
    ) == check.normalize_requirement("pyarrow<26,>=24")
    assert (
        check.normalize_requirement('sglang==0.5.18; extra == "thead"')
        == "sglang==0.5.18"
    )


def test_tests_payload_fails(check):
    assert _run(check, _make_wheel(with_tests=True)) == 1


def test_missing_import_guard_payload_fails(check):
    reduced = [p for p in VALID_PAYLOAD if not p.endswith("import_guard.py")]
    assert _run(check, _make_wheel(payload=reduced)) == 1


def test_version_mismatch_fails(check):
    assert _run(check, _make_wheel(version="9.9.9")) == 1
