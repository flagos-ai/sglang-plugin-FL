"""Unit tests for the precise pip-check exception parser (Hygon)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docker/hygon/scripts"))

from pip_check_exceptions import (  # noqa: E402
    ALLOWED_NEW_REQUIREMENTS,
    PIP_CHECK_SUCCESS_NOTICE,
    pip_check_violations,
    requirement_distribution_name,
)

BASELINE: set[str] = set()


class RequirementNameTests(unittest.TestCase):
    def test_version_specifiers_reduce_to_distribution_name(self):
        self.assertEqual(requirement_distribution_name("triton>=3.5"), "triton")
        self.assertEqual(requirement_distribution_name("triton == 3.8.0"), "triton")
        # Raw names pass through; canonicalization happens at pair time.
        self.assertEqual(requirement_distribution_name("XGrammar"), "XGrammar")


class PipCheckExceptionTests(unittest.TestCase):
    def test_xgrammar_triton_is_accepted(self):
        line = "xgrammar 0.2.1 requires triton, which is not installed."
        self.assertEqual(pip_check_violations(1, line + "\n", "", BASELINE), set())

    def test_any_other_requirer_of_triton_is_rejected(self):
        line = "tilelang 0.1.0 requires triton, which is not installed."
        self.assertEqual(pip_check_violations(1, line + "\n", "", BASELINE), {line})

    def test_xgrammar_requiring_something_else_is_rejected(self):
        line = "xgrammar 0.2.1 requires torch, which is not installed."
        self.assertEqual(pip_check_violations(1, line + "\n", "", BASELINE), {line})

    def test_similar_names_are_rejected(self):
        lines = [
            "xgrammars 0.2.1 requires triton, which is not installed.",
            "xgrammar 0.2.1 requires tritons, which is not installed.",
            "xgrammar-extra 0.2.1 requires triton3, which is not installed.",
        ]
        stdout = "\n".join(lines) + "\n"
        self.assertEqual(pip_check_violations(1, stdout, "", BASELINE), set(lines))

    def test_version_constrained_triton_matches_by_distribution_name(self):
        line = "xgrammar 0.2.1 requires triton>=3.5, which is not installed."
        self.assertEqual(pip_check_violations(1, line + "\n", "", BASELINE), set())
        constrained_other = (
            "xgrammar 0.2.1 requires tritonx>=3.5, which is not installed."
        )
        self.assertEqual(
            pip_check_violations(1, constrained_other + "\n", "", BASELINE),
            {constrained_other},
        )

    def test_unknown_output_format_is_rejected(self):
        unknown = "some package has a conflict with another package"
        self.assertEqual(
            pip_check_violations(1, unknown + "\n", "", BASELINE), {unknown}
        )
        empty_line = ""
        self.assertEqual(pip_check_violations(1, empty_line, "", BASELINE), set())

    def test_crashing_pip_check_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "returncode=2.*boom"):
            pip_check_violations(2, "partial stdout", "boom", BASELINE)

    def test_success_notice_and_baseline_entries_are_ignored(self):
        baseline = {"runtimepkg 1.0 requires olddep, which is not installed."}
        stdout = (
            f"{PIP_CHECK_SUCCESS_NOTICE}\n"
            "runtimepkg 1.0 requires olddep, which is not installed.\n"
            "xgrammar 0.2.1 requires triton, which is not installed.\n"
        )
        self.assertEqual(pip_check_violations(1, stdout, "", baseline), set())

    def test_allowed_pairs_contract(self):
        self.assertEqual(ALLOWED_NEW_REQUIREMENTS, frozenset({("xgrammar", "triton")}))


if __name__ == "__main__":
    unittest.main()
