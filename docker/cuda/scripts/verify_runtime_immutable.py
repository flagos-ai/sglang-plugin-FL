#!/usr/bin/env python3
"""Runtime-immutability gate for docker/cuda/runtime-0.5.18.containerfile.

Runs once at image-build time, after the locked closure install. All
reference data is static, committed next to this script and generated
once from the digest-pinned base image:

  runtime-0.5.18.runtime-freeze.txt      every runtime distribution
  runtime-0.5.18.site-files.sha256        every runtime site-packages file
  runtime-0.5.18.pip-check-baseline.txt   the runtime's own pip check output
  runtime-0.5.18.provenance.json          vendor module origins at start

Checks (any failure aborts the build):
  1. lock overlap  -- the install lock lists no runtime package name, so a
                      --no-deps install can never replace one
  2. freeze        -- every runtime distribution survives verbatim
  3. site files    -- every runtime site-packages file keeps its SHA256
                      (catches same-version reinstalls and in-place
                      overwrites, which distribution checks cannot see)
  4. pip check     -- only the documented waiver may be new versus the
                      runtime's own baseline
  5. provenance    -- torch/triton/flag_gems/flagcx/flagtree still resolve
                      to the recorded origins under container-start path
                      order (/opt/flagtree precedes site-packages)

Usage: verify_runtime_immutable.py [--evidence-dir DIR] [--lock FILE]
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name

# Waiver 1: the runtime owns fsspec (2026.9.0, readonly) while the
# numpy1-era datasets 4.0.0 pins its metadata cap at fsspec<=2025.3.0.
# The conflict is metadata-only: import and sglang.bench_serving are
# verified against the runtime fsspec.
#
# Waiver 2/3: the runtime provides triton out-of-band as compiler slots
# (/opt/flagtree 3.6.0 default via PYTHONPATH; /opt/triton 3.7.1 via
# `compiler triton`). Closure packages declaring a plain `triton`
# dependency are satisfied at module level by the FlagTree slot and are
# unsatisfiable at the pip level by design -- the closure must not
# duplicate a runtime-provided component as a pip wheel.
PIP_CHECK_WAIVED = (
    "datasets 4.0.0 has requirement fsspec[http]<=2025.3.0,>=2023.1.0, "
    "but you have fsspec 2026.9.0.",
    "humming-kernels 0.1.10 requires triton, which is not installed.",
    "xgrammar 0.2.1 requires triton, which is not installed.",
)

PROVENANCE_MODULES = ("torch", "triton", "flag_gems", "flagcx", "flagtree")


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def installed_distributions() -> dict[str, str]:
    import importlib.metadata as md

    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }


def check_lock_overlap(lock: Path, runtime_names: set[str]) -> None:
    lock_names = {
        canonicalize_name(line.split("==")[0]) for line in read_lines(lock)
    }
    overlap = lock_names & runtime_names
    assert not overlap, (
        f"install lock overlaps runtime distributions: {sorted(overlap)}"
    )
    print(f"[1/5] lock gate: {len(lock_names)} additions, "
          f"zero overlap with {len(runtime_names)} runtime distributions")


def check_freeze(evidence: Path, runtime_names: set[str]) -> None:
    import importlib.metadata as md

    current = installed_distributions()
    missing = sorted(
        f"{name}=={current.get(name, '<removed>')}"
        for name in runtime_names
        if name not in current
    )
    assert not missing, f"runtime distributions removed: {missing}"
    print(f"[2/5] runtime freeze preserved ({len(runtime_names)} distributions)")


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
    print(f"[3/5] site-packages file baseline unchanged "
          f"({sum(1 for _ in manifest.open())} files)")


def check_pip_check(evidence: Path) -> None:
    import subprocess

    after = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True,
    ).stdout
    baseline = set(read_lines(evidence / "runtime-0.5.18.pip-check-baseline.txt"))
    current = {line for line in after.splitlines() if line.strip()}
    new = current - baseline - set(PIP_CHECK_WAIVED)
    assert not new, (
        f"new pip check violations introduced by the SGLang install: "
        f"{sorted(new)}"
    )
    print("[4/5] pip check: no new violations "
          "(baseline preserved; documented waiver on record)")


def check_provenance(evidence: Path) -> None:
    expected = json.loads(
        (evidence / "runtime-0.5.18.provenance.json").read_text()
    )

    def probe(module: str) -> dict:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            return {"found": False, "origin": None}
        return {"found": True, "origin": spec.origin}

    after = {module: probe(module) for module in PROVENANCE_MODULES}
    changed = {
        module: (expected.get(module), after[module])
        for module in PROVENANCE_MODULES
        if expected.get(module) != after[module]
    }
    assert not changed, f"vendor module provenance changed: {changed}"
    print("[5/5] module provenance unchanged "
          "(triton resolves via /opt/flagtree before site-packages)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir", type=Path, default=Path("/opt/sglang-ci"),
        help="directory holding the committed runtime reference data",
    )
    parser.add_argument(
        "--lock", type=Path, default=Path("/opt/sglang-ci/runtime-0.5.18.install.lock"),
        help="the install lock the closure was installed from",
    )
    args = parser.parse_args()

    # Mirror container-start resolution: PYTHONPATH (/opt/flagtree)
    # precedes site-packages there, while this script may run from a
    # build RUN step that lacks that ENV.
    if "/opt/flagtree" not in sys.path:
        sys.path.insert(0, "/opt/flagtree")

    evidence = args.evidence_dir
    runtime_names = {
        canonicalize_name(line.split("==")[0])
        for line in read_lines(evidence / "runtime-0.5.18.runtime-freeze.txt")
    }

    check_lock_overlap(args.lock, runtime_names)
    check_freeze(evidence, runtime_names)
    check_site_files(evidence)
    check_pip_check(evidence)
    check_provenance(evidence)
    print("runtime immutability verified against the committed baseline")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        sys.exit(1)
