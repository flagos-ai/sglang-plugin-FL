#!/usr/bin/env python3
"""Precise pip-check exception parsing for the Hygon two-wheel CI image.

The image resolves the application closure without a standalone ``triton``
distribution (docker/hygon/runtime-0.5.18.excludes.txt): the Runtime
provides the Triton compiler as the /opt/flagtree module tree. Wheel
metadata that still declares triton surfaces in ``pip check`` as a
missing-distribution line, and ONLY the exact canonical
(requirer, requirement) pairs in ``ALLOWED_NEW_REQUIREMENTS`` are
accepted as documented platform capability exceptions. Everything else -
unknown line shapes included - is a violation; a crashing pip check
fails loudly with its returncode, stdout and stderr.

Shared by the image build gate (docker/hygon/runtime-0.5.18.containerfile),
the post-build verifier (docker/hygon/scripts/verify_runtime_immutable.py)
and the CI-time environment verification.
"""

from __future__ import annotations

import re

from packaging.utils import canonicalize_name

PIP_CHECK_SUCCESS_NOTICE = "No broken requirements found."

# (requirer, requirement) pairs the image may add on top of the runtime's
# own pip-check baseline, exactly and completely. xgrammar's wheel metadata
# declares triton on the Linux/x86_64 platform; the compiler comes from the
# Runtime's /opt/flagtree tree instead. This is a documented platform
# capability exception, not a satisfied packaging contract.
ALLOWED_NEW_REQUIREMENTS = frozenset(
    {
        ("xgrammar", "triton"),
    }
)

# pip check reports a missing distribution as:
#   <requirer> <version> requires <requirement>, which is not installed.
_PIP_CHECK_MISSING_RE = re.compile(
    r"^(?P<requirer>\S+) \S+ requires (?P<requirement>[^,]+), "
    r"which is not installed\.$"
)


def requirement_distribution_name(requirement: str) -> str:
    """Distribution name of a requirement string ('triton>=3.5' -> 'triton')."""
    return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0]


def pip_check_violations(
    returncode: int,
    stdout: str,
    stderr: str,
    baseline: set[str],
    allowed_pairs: frozenset[tuple[str, str]] = ALLOWED_NEW_REQUIREMENTS,
) -> set[str]:
    """New pip-check violations, or an error when pip check itself failed.

    A new finding is accepted only when it parses as a missing-distribution
    line whose canonical (requirer, requirement-name) pair is in
    ``allowed_pairs``; everything else - unknown line shapes included - is
    a violation. Returncode 0 means clean; anything other than 0 or 1 is a
    crashing pip check and raises with its full output.
    """
    if returncode == 0:
        # Clean run: stdout carries the success notice, not violations.
        return set()
    if returncode != 1:
        raise RuntimeError(
            "pip check itself failed: "
            f"returncode={returncode} stdout={stdout!r} stderr={stderr!r}"
        )
    current = {
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and line.strip() != PIP_CHECK_SUCCESS_NOTICE
    }
    violations: set[str] = set()
    for line in current - baseline:
        match = _PIP_CHECK_MISSING_RE.match(line)
        if match is None:
            violations.add(line)
            continue
        pair = (
            canonicalize_name(match.group("requirer")),
            canonicalize_name(
                requirement_distribution_name(match.group("requirement"))
            ),
        )
        if pair not in allowed_pairs:
            violations.add(line)
    return violations
