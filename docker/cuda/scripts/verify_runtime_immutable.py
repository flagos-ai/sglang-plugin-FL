#!/usr/bin/env python3
"""Runtime-immutability baseline generator and verifier for the NVIDIA CI image.

One script, two modes, so the generated baseline and its consumer always
share the same format logic:

``--generate --out DIR``  run INSIDE the Runtime base image (docker run
  from the digest pinned in the containerfile -- never inside the
  candidate) and write the reference data:

    runtime-freeze.txt      distributions (name==version)
    site-files.sha256       site-packages file hashes
    pip-check-baseline.txt  the base's own pip check findings
    provenance.json         compiler slot versions + module origins

  These are derived data, not source: they live in a temporary directory
  (or CI artifacts), are regenerated whenever the Runtime base digest
  changes, and are never committed to the repository.

``--verify --evidence-dir DIR``  run INSIDE the completed candidate image
  against a generated baseline:

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
        (evidence / "runtime-freeze.txt").read_text().splitlines()
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
    manifest = evidence / "site-files.sha256"
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
        for line in (evidence / "pip-check-baseline.txt").read_text().splitlines()
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
    recorded = json.loads((evidence / "provenance.json").read_text())
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
        assert expected_version, f"the baseline provenance lacks the version for {key}"
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


# --- baseline generation (run inside the pinned Runtime base) --------------
# Derived data only: regenerated per base digest, never committed.

PROVENANCE_MODULES = (
    "torch",
    "flag_gems",
    "flagcx",
    "flagtree",
    "triton",
)
COMPILER_SLOT_INITS = {
    "/opt/flagtree/triton": Path("/opt/flagtree/triton/__init__.py"),
    "/opt/triton/triton": Path("/opt/triton/triton/__init__.py"),
}


def _write_freeze(out: Path) -> None:
    lines = sorted(
        f"{name}=={version}" for name, version in installed_distributions().items()
    )
    (out / "runtime-freeze.txt").write_text("\n".join(lines) + "\n")
    print(f"[generate] runtime-freeze.txt: {len(lines)} distributions")


def _write_site_files(out: Path) -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    digest = hashlib.sha256()
    count = 0
    with open(out / "site-files.sha256", "w") as manifest:
        for path in sorted(purelib.rglob("*")):
            if not path.is_file():
                continue
            file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write(f"{file_digest}  {path.relative_to(purelib)}\n")
            digest.update(file_digest.encode())
            count += 1
    print(
        f"[generate] site-files.sha256: {count} files (digest {digest.hexdigest()[:12]}…)"
    )


def _write_pip_check(out: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "pip check itself failed during baseline generation: "
            f"returncode={completed.returncode} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    findings = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and line.strip() != PIP_CHECK_SUCCESS_NOTICE
    )
    (out / "pip-check-baseline.txt").write_text(
        "\n".join(findings) + ("\n" if findings else "")
    )
    print(f"[generate] pip-check-baseline.txt: {len(findings)} baseline findings")


def _write_provenance(out: Path) -> None:
    import importlib.util

    provenance: dict[str, object] = {"compiler_slots": {}}
    for slot, init_path in COMPILER_SLOT_INITS.items():
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", init_path.read_text())
        assert match, f"cannot read the version from {init_path}"
        provenance["compiler_slots"][slot] = match.group(1)
    for module_name in PROVENANCE_MODULES:
        spec = importlib.util.find_spec(module_name)
        provenance[module_name] = {
            "found": spec is not None and bool(spec.origin),
            "origin": spec.origin if spec else None,
        }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True)
    )
    print("[generate] provenance.json: compiler slots + module origins")


def generate_baseline(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _write_freeze(out)
    _write_site_files(out)
    _write_pip_check(out)
    _write_provenance(out)
    print(f"runtime baseline generated in {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate",
        action="store_true",
        help="write the runtime baseline (--out) from THIS environment; "
        "run inside the pinned Runtime base image",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify THIS environment against the baseline in --evidence-dir; "
        "run inside the candidate image",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="directory to write the generated baseline into",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="bind-mounted directory with the generated runtime baseline",
    )
    args = parser.parse_args()

    try:
        if args.generate:
            if args.out is None:
                parser.error("--generate requires --out")
            generate_baseline(args.out)
            return 0
        if args.evidence_dir is None:
            parser.error("--verify requires --evidence-dir")
        evidence = args.evidence_dir
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
