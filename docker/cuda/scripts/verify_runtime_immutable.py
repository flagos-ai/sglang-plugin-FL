#!/usr/bin/env python3
"""Post-build runtime-immutability verifier for the NVIDIA CI image.

Runs INSIDE the completed image (docker run --rm --volume
<docker/cuda>:/evidence:ro --entrypoint python3 <image>
/evidence/scripts/verify_runtime_immutable.py --evidence-dir /evidence),
never during image construction. All reference data is bind-mounted:

  runtime-0.5.18.runtime-freeze.txt      runtime distributions (name==version)
  runtime-0.5.18.site-files.sha256       runtime site-packages file hashes
  runtime-0.5.18.pip-check-baseline.txt  the runtime's own pip check findings
  runtime-0.5.18.provenance.json         compiler slot inventory

Checks:
  1. distribution freeze -- every runtime distribution survives with the
     exact version string, local versions included (torch==2.13.0+cu130);
     any upgrade, downgrade, or removal fails
  2. site files          -- every runtime site-packages file keeps its SHA256
  3. pip check           -- returncode-aware comparison against the runtime
     baseline; the success notice is never treated as a violation and a
     crashing pip check fails loudly with returncode, stdout, and stderr
     in the report. The only accepted additions are the exact
     (requirer, triton) pairs in ALLOWED_NEW_REQUIREMENTS: the image
     resolves without a standalone triton distribution while
     Torch/xgrammar/humming-kernels still declare it in their metadata --
     a documented platform capability exception (the compiler is the
     Runtime's /opt/flagtree module tree). Matching is exact on the
     canonical requirer and requirement names, never substring-based.
  4. compiler slots      -- /opt/flagtree and /opt/triton still carry the
     recorded Triton versions

It also prints the additions manifest (installed distributions that are not
runtime distributions) as an audit trail; it is output, never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name

PIP_CHECK_SUCCESS_NOTICE = "No broken requirements found."

# (requirer, requirement) pairs that the image may add on top of the
# runtime baseline, exactly and completely. The two-wheel install resolves
# without a standalone triton distribution; these requirers still declare
# triton in their wheel metadata, and the compiler comes from the
# Runtime's /opt/flagtree module tree instead. This is a documented
# platform capability exception, not a satisfied packaging contract.
ALLOWED_NEW_REQUIREMENTS = frozenset(
    {
        ("xgrammar", "triton"),
        ("humming-kernels", "triton"),
    }
)

# pip check reports a missing distribution as:
#   <requirer> <version> requires <requirement>, which is not installed.
_PIP_CHECK_MISSING_RE = re.compile(
    r"^(?P<requirer>\S+) \S+ requires (?P<requirement>[^,]+), "
    r"which is not installed\.$"
)


def parse_freeze_lines(lines: list[str]) -> dict[str, str]:
    """Map canonical name -> full version string from ``pip freeze`` lines.

    Local versions are preserved verbatim (``torch==2.13.0+cu130``); lines
    without an ``==`` (editables, VCS references) are not runtime freeze
    entries and are skipped.
    """
    frozen: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        frozen[canonicalize_name(name)] = version
    return frozen


def diff_runtime(
    baseline: dict[str, str], current: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Return (missing, version_changed) against the runtime baseline.

    A runtime distribution appearing with any other version string -- even
    the same release without its local label -- counts as changed.
    """
    missing = sorted(name for name in baseline if name not in current)
    version_changed = sorted(
        f"{name}: {baseline[name]} -> {current[name]}"
        for name in baseline
        if name in current and current[name] != baseline[name]
    )
    return missing, version_changed


def pip_check_violations(
    returncode: int,
    stdout: str,
    stderr: str,
    baseline: set[str],
    allowed_pairs: frozenset[tuple[str, str]] = ALLOWED_NEW_REQUIREMENTS,
) -> set[str]:
    """New pip check violations, or errors when pip check itself failed.

    A new finding is accepted only when it parses as a missing-distribution
    line whose canonical (requirer, requirement-name) pair is in
    ``allowed_pairs``; everything else -- unknown line shapes included --
    is a violation.
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
        requirement = re.split(
            r"[\s\[<>=!~;]", match.group("requirement").strip(), maxsplit=1
        )[0]
        pair = (
            canonicalize_name(match.group("requirer")),
            canonicalize_name(requirement),
        )
        if pair not in allowed_pairs:
            violations.add(line)
    return violations


def installed_distributions() -> dict[str, str]:
    import importlib.metadata as md

    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }


def check_freeze(evidence: Path) -> set[str]:
    baseline = parse_freeze_lines(
        (evidence / "runtime-0.5.18.runtime-freeze.txt").read_text().splitlines()
    )
    current = installed_distributions()
    missing, version_changed = diff_runtime(baseline, current)
    assert not missing, f"runtime distributions removed: {missing}"
    assert not version_changed, (
        f"runtime distribution versions changed: {version_changed}"
    )
    print(
        f"[1/4] runtime distributions preserved: {len(baseline)} (exact name+version)"
    )
    return set(baseline)


def check_site_files(evidence: Path) -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    manifest = evidence / "runtime-0.5.18.site-files.sha256"
    missing, changed = [], []
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = purelib / relative
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    assert not missing and not changed, (
        f"site-packages drifted: {len(missing)} missing, "
        f"{len(changed)} changed; first missing {missing[:3]}, "
        f"first changed {changed[:3]}"
    )
    print("[2/4] site-packages file baseline unchanged")


def check_pip_check(evidence: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    baseline = {
        line.strip()
        for line in (evidence / "runtime-0.5.18.pip-check-baseline.txt")
        .read_text()
        .splitlines()
        if line.strip() and line.strip() != PIP_CHECK_SUCCESS_NOTICE
    }
    new = pip_check_violations(
        completed.returncode, completed.stdout, completed.stderr, baseline
    )
    assert not new, (
        f"new pip check violations introduced by the two-wheel install "
        f"(only the exact triton requirers {sorted(ALLOWED_NEW_REQUIREMENTS)} "
        f"are accepted): {sorted(new)}"
    )
    accepted = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and line.strip() != PIP_CHECK_SUCCESS_NOTICE
    }
    for line in sorted(accepted - baseline):
        print(f"[3/4] accepted (metadata-only triton gap): {line}")
    print(
        "[3/4] pip check: only the documented platform capability exception "
        "versus the runtime baseline"
    )


def check_compiler_slots(evidence: Path) -> None:
    recorded = json.loads((evidence / "runtime-0.5.18.provenance.json").read_text())
    slots = recorded.get("compiler_slots", {})
    expected = {
        "flagtree_slot": (
            Path("/opt/flagtree/triton/__init__.py"),
            slots.get("/opt/flagtree/triton"),
        ),
        "triton_slot": (
            Path("/opt/triton/triton/__init__.py"),
            slots.get("/opt/triton/triton"),
        ),
    }
    for key, (init_path, expected_version) in expected.items():
        assert init_path.is_file(), f"compiler slot missing: {init_path}"
        assert expected_version, f"provenance.json lacks the version for {key}"
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", init_path.read_text())
        actual = match.group(1) if match else None
        assert actual == expected_version, (
            f"compiler slot {init_path} version drifted: "
            f"expected {expected_version}, found {actual}"
        )
        print(f"[4/4] {key}: triton {actual} at {init_path.parent}")


def report_additions(runtime_names: set[str]) -> None:
    current = installed_distributions()
    additions = sorted(
        f"{name}=={current[name]}" for name in current if name not in runtime_names
    )
    print(f"[audit] {len(additions)} distributions added on top of the runtime:")
    for line in additions:
        print(f"[audit]   {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="bind-mounted directory with the runtime reference data",
    )
    args = parser.parse_args()
    evidence = args.evidence_dir

    try:
        runtime_names = check_freeze(evidence)
        check_site_files(evidence)
        check_pip_check(evidence)
        check_compiler_slots(evidence)
        report_additions(runtime_names)
    except AssertionError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    except RuntimeError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print("runtime immutability verified against the bind-mounted baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
