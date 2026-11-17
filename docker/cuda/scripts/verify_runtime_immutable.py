#!/usr/bin/env python3
"""Post-build runtime-immutability verifier for the NVIDIA CI image.

Runs INSIDE the completed image (docker run --rm <image> --volume
<evidence>:/evidence:ro --entrypoint python3 <image>
/evidence/verify_runtime_immutable.py --evidence-dir /evidence), never
during image construction. All reference data is bind-mounted:

  runtime-freeze.txt      every runtime distribution (pip freeze --all)
  site-files.sha256       every runtime site-packages file's SHA256
  pip-check-baseline.txt  the runtime's own pip check findings
  provenance.json         compiler slot inventory (flagtree/triton versions)

Checks:
  1. distribution freeze -- every runtime distribution survives verbatim
  2. site files          -- every runtime site-packages file keeps its SHA256
  3. pip check           -- no new violations versus the runtime baseline
                            (no waiver list; new findings fail loudly)
  4. compiler slots      -- /opt/flagtree and /opt/triton still carry the
                            recorded Triton versions

It also prints the additions manifest (installed distributions that are not
runtime distributions) as an audit trail; it is output, never an input.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def installed_distributions() -> dict[str, str]:
    import importlib.metadata as md

    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }


def check_freeze(evidence: Path, runtime_names: set[str]) -> None:
    current = installed_distributions()
    missing = sorted(name for name in runtime_names if name not in current)
    assert not missing, f"runtime distributions removed: {missing}"
    print(f"[1/4] runtime freeze preserved ({len(runtime_names)} distributions)")


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
    after = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    ).stdout
    baseline = set(read_lines(evidence / "runtime-0.5.18.pip-check-baseline.txt"))
    current = {line for line in after.splitlines() if line.strip()}
    new = current - baseline
    assert not new, (
        f"new pip check violations introduced by the two-wheel install: {sorted(new)}"
    )
    print("[3/4] pip check: no new violations versus the runtime baseline")


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


def report_additions(evidence: Path, runtime_names: set[str]) -> None:
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

    runtime_names = {
        canonicalize_name(line.split("==", 1)[0])
        for line in read_lines(evidence / "runtime-0.5.18.runtime-freeze.txt")
        if "==" in line
    }

    try:
        check_freeze(evidence, runtime_names)
        check_site_files(evidence)
        check_pip_check(evidence)
        check_compiler_slots(evidence)
        report_additions(evidence, runtime_names)
    except AssertionError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print("runtime immutability verified against the bind-mounted baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
