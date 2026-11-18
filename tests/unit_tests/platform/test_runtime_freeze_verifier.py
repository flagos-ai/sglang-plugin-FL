# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the post-build runtime verifier's pure logic.

Loads docker/cuda/scripts/verify_runtime_immutable.py by path (it is a
build-side tool, not an installed package) and pins the two contracts the
acceptance review found broken:

* the runtime freeze comparison must be version-exact, local versions
  included -- a name-only check cannot see upgrades or downgrades;
* pip check parsing must be returncode-aware -- the success notice is not
  a violation, and a crashing pip check must not silently pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "cuda"
    / "scripts"
    / "verify_runtime_immutable.py"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("runtime_freeze_verifier", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _load_verifier()


BASELINE = {
    "torch": "2.13.0+cu130",
    "numpy": "1.26.4",
    "fsspec": "2026.9.0",
}


class TestParseFreezeLines:
    def test_name_version_and_local_version_preserved(self, verifier):
        parsed = verifier.parse_freeze_lines(
            ["torch==2.13.0+cu130", "PyYAML==6.0.1", ""]
        )
        assert parsed == {"torch": "2.13.0+cu130", "pyyaml": "6.0.1"}

    def test_non_freeze_lines_skipped(self, verifier):
        parsed = verifier.parse_freeze_lines(
            ["-e git+https://example.invalid/x#egg=x", "# comment", "numpy==1.26.4"]
        )
        assert parsed == {"numpy": "1.26.4"}


class TestDiffRuntime:
    def test_identical_passes(self, verifier):
        missing, changed = verifier.diff_runtime(BASELINE, dict(BASELINE))
        assert missing == []
        assert changed == []

    def test_missing_fails(self, verifier):
        current = dict(BASELINE)
        del current["numpy"]
        missing, _ = verifier.diff_runtime(BASELINE, current)
        assert missing == ["numpy"]

    def test_upgrade_fails(self, verifier):
        current = dict(BASELINE, numpy="1.27.0")
        _, changed = verifier.diff_runtime(BASELINE, current)
        assert changed == ["numpy: 1.26.4 -> 1.27.0"]

    def test_downgrade_fails(self, verifier):
        current = dict(BASELINE, numpy="1.25.0")
        _, changed = verifier.diff_runtime(BASELINE, current)
        assert changed == ["numpy: 1.26.4 -> 1.25.0"]

    def test_local_version_change_fails(self, verifier):
        current = dict(BASELINE, torch="2.13.0")
        _, changed = verifier.diff_runtime(BASELINE, current)
        assert changed == ["torch: 2.13.0+cu130 -> 2.13.0"]
        current = dict(BASELINE, torch="2.13.0+cu131")
        _, changed = verifier.diff_runtime(BASELINE, current)
        assert changed == ["torch: 2.13.0+cu130 -> 2.13.0+cu131"]

    def test_new_distribution_does_not_affect_baseline(self, verifier):
        current = {**BASELINE, "sglang": "0.5.18"}
        missing, changed = verifier.diff_runtime(BASELINE, current)
        assert missing == []
        assert changed == []


class TestPipCheckViolations:
    def test_clean_run_has_no_violations(self, verifier):
        assert (
            verifier.pip_check_violations(
                0, "No broken requirements found.\n", "", set()
            )
            == set()
        )

    def test_baseline_only_problems_pass(self, verifier):
        baseline = {"torch 2.13.0+cu130 requires triton, which is not installed."}
        stdout = "torch 2.13.0+cu130 requires triton, which is not installed.\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == set()

    def test_new_problem_fails(self, verifier):
        baseline = {"old problem"}
        stdout = "old problem\nbrand new problem\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == {
            "brand new problem"
        }

    def test_crashing_pip_check_reports_stdout_and_stderr(self, verifier):
        with pytest.raises(RuntimeError) as excinfo:
            verifier.pip_check_violations(
                2, "partial stdout", "pip internals exploded", set()
            )
        message = str(excinfo.value)
        assert "returncode=2" in message
        assert "partial stdout" in message
        assert "pip internals exploded" in message

    def test_documented_triton_exception_is_the_only_accepted_addition(
        self, verifier
    ):
        baseline = {"torch 2.13.0+cu130 requires triton, which is not installed."}
        stdout = (
            "torch 2.13.0+cu130 requires triton, which is not installed.\n"
            "xgrammar 0.8.2 requires triton, which is not installed.\n"
            "humming-kernels 0.4.1 requires triton, which is not installed.\n"
        )
        assert verifier.pip_check_violations(1, stdout, "", baseline) == set()

    def test_other_triton_requirer_is_a_violation(self, verifier):
        baseline = {"torch 2.13.0+cu130 requires triton, which is not installed."}
        stdout = "flashinfer-python 0.6.17 requires triton, which is not installed.\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == {
            "flashinfer-python 0.6.17 requires triton, which is not installed."
        }

    def test_versioned_requirement_matches_by_distribution_name(self, verifier):
        baseline = set()
        stdout = "xgrammar 0.8.2 requires triton>=3.2, which is not installed.\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == set()

    def test_other_requirement_of_an_allowed_requirer_is_a_violation(self, verifier):
        baseline = set()
        stdout = "xgrammar 0.8.2 requires pydantic, which is not installed.\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == {
            "xgrammar 0.8.2 requires pydantic, which is not installed."
        }

    def test_exact_pairs_only_no_substring_acceptance(self, verifier):
        # A requirer whose canonical name merely contains an allowed name
        # must not inherit the exception.
        baseline = set()
        stdout = "xgrammar-extra 1.0 requires triton, which is not installed.\n"
        assert verifier.pip_check_violations(1, stdout, "", baseline) == {
            "xgrammar-extra 1.0 requires triton, which is not installed."
        }
