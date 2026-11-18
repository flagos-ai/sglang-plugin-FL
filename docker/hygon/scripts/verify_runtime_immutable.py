#!/usr/bin/env python3
"""Post-build runtime-immutability verifier for the Hygon two-wheel CI image.

Runs INSIDE the completed image (docker/hygon/scripts/verify_image.sh wraps
the docker run), never during image construction. All runtime reference
data is bind-mounted read-only:

  runtime-0.5.18.runtime-freeze.txt      runtime distributions (name==version)
  runtime-0.5.18.site-files.sha256       runtime site-packages file hashes
  runtime-0.5.18.pip-check-baseline.txt  the runtime's own pip check findings
  runtime-0.5.18.provenance.json         compiler slot inventory

Checks:
  1. distribution freeze -- every runtime distribution survives with the
     exact version string, local versions included; any upgrade,
     downgrade, or removal fails
  2. site files          -- every runtime site-packages file keeps its SHA256
  3. pip check           -- returncode-aware comparison against the runtime
     baseline; the ONLY accepted addition is the exact canonical
     (xgrammar, triton) pair, matched by the shared precise parser
     (pip_check_exceptions.py), never substring-based
  4. compiler slots      -- /opt/flagtree carries the recorded Triton, no
     standalone triton distribution exists, and the triton module
     resolves from /opt/flagtree in THIS process
  5. two-wheel contract  -- sglang and sglang_fl distributions, versions,
     entry points, and the plugin resolving from its installed
     distribution root
  6. HCU compat markers  -- the clang wrapper + ENV, the stub finder with
     the scalar_type exemption, and the applied HCU patch record

It also prints the additions manifest as an audit trail; it is output,
never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pip_check_exceptions import pip_check_violations  # noqa: E402

EXPECTED_SGLANG = "0.5.18+flagos"
EXPECTED_PLUGIN = "0.1.0"
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": "sglang_fl",
    "sglang.srt.plugins": "sglang_fl",
}
HCU_PATCH_RECORD = "/opt/sglang-ci/sglang-hcu-patch.sha256"
STUB_FINDER = "site-packages/_hcu_kernel_stub_finder.py"
CLANG_WRAPPER = Path("/opt/hcu-clang/clang-18-device-libs")


def installed_distributions() -> dict[str, str]:
    import importlib.metadata as md

    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }


def parse_freeze_lines(lines: list[str]) -> dict[str, str]:
    frozen: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        frozen[canonicalize_name(name)] = version
    return frozen


def check_freeze(evidence: Path) -> set[str]:
    baseline = parse_freeze_lines(
        (evidence / "runtime-0.5.18.runtime-freeze.txt").read_text().splitlines()
    )
    current = installed_distributions()
    missing = sorted(name for name in baseline if name not in current)
    version_changed = sorted(
        f"{name}: {baseline[name]} -> {current[name]}"
        for name in baseline
        if name in current and current[name] != baseline[name]
    )
    assert not missing, f"runtime distributions removed: {missing}"
    assert not version_changed, (
        f"runtime distribution versions changed: {version_changed}"
    )
    print(f"[1/6] runtime distributions preserved: {len(baseline)} (exact)")
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
    print("[2/6] site-packages file baseline unchanged")


def check_pip_check(evidence: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    baseline = {
        line.strip()
        for line in (evidence / "runtime-0.5.18.pip-check-baseline.txt")
        .read_text()
        .splitlines()
        if line.strip() and line.strip() != "No broken requirements found."
    }
    new = pip_check_violations(
        completed.returncode, completed.stdout, completed.stderr, baseline
    )
    assert not new, (
        f"new pip check violations beyond the platform exception: {sorted(new)}"
    )
    print("[3/6] pip check: only the documented (xgrammar, triton) exception")


def check_compiler_slots(evidence: Path) -> None:
    recorded = json.loads((evidence / "runtime-0.5.18.provenance.json").read_text())
    slots = recorded.get("compiler_slots", {})
    for slot, expected_version in slots.items():
        init_path = Path(slot) / "triton" / "__init__.py"
        if slot == "/opt/flagtree":
            assert init_path.is_file(), f"compiler slot missing: {init_path}"
            match = re.search(
                r"__version__\s*=\s*['\"]([^'\"]+)", init_path.read_text()
            )
            actual = match.group(1) if match else None
            assert actual == expected_version, (
                f"compiler slot {slot} drifted: expected {expected_version}, "
                f"found {actual}"
            )
            print(f"[4/6] flagtree slot: triton {actual} at {slot}/triton")
    current = installed_distributions()
    assert "triton" not in current, (
        f"standalone triton distribution present: {current['triton']}"
    )
    module = importlib.import_module("triton")
    assert str(module.__file__).startswith("/opt/flagtree/"), (
        f"triton module resolved {module.__file__}, expected /opt/flagtree"
    )
    print(f"[4/6] no standalone triton; module resolves {module.__file__}")


def check_two_wheel_contract() -> None:
    import importlib.metadata as md

    assert md.version("sglang") == EXPECTED_SGLANG, md.version("sglang")
    assert md.version("sglang-fl") == EXPECTED_PLUGIN, md.version("sglang-fl")
    for group, target in EXPECTED_ENTRY_POINTS.items():
        registered = sorted(
            ep.value for ep in md.entry_points(group=group) if ep.name == target
        )
        assert registered == [target], f"{group}: {registered}"
    plugin = importlib.import_module("sglang_fl")
    from importlib import metadata

    distribution_root = Path(
        metadata.distribution("sglang-fl").locate_file("")
    ).resolve()
    module_file = Path(plugin.__file__).resolve()
    assert module_file.is_relative_to(distribution_root), (
        f"sglang_fl at {module_file} is not from the installed distribution "
        f"root {distribution_root}"
    )
    attention = importlib.import_module(
        "sglang_fl.dispatch.backends.vendor.hygon.impl.attention_backend"
    )
    assert str(Path(attention.__file__)).startswith(str(distribution_root)), (
        f"attention backend outside the wheel: {attention.__file__}"
    )
    print(
        f"[5/6] two-wheel contract: sglang=={EXPECTED_SGLANG}, "
        f"sglang_fl=={EXPECTED_PLUGIN} from {distribution_root}"
    )


def check_hcu_compat_markers() -> None:
    wrapper_env = os.environ.get("TRITON_HIP_CLANG_PATH", "")
    assert CLANG_WRAPPER.is_file(), f"clang wrapper missing: {CLANG_WRAPPER}"
    assert wrapper_env == str(CLANG_WRAPPER), (
        f"TRITON_HIP_CLANG_PATH={wrapper_env!r} != {CLANG_WRAPPER}"
    )
    finder = Path(sysconfig.get_paths()["purelib"]) / "_hcu_kernel_stub_finder.py"
    text = finder.read_text()
    assert 'fullname == "sgl_kernel.scalar_type"' in text, (
        "stub finder lacks the scalar_type exemption"
    )
    assert Path(HCU_PATCH_RECORD).is_file(), f"HCU patch record missing"
    print(
        "[6/6] HCU compat markers: clang wrapper + scalar_type fallback + patch record"
    )


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
        check_two_wheel_contract()
        check_hcu_compat_markers()
        report_additions(runtime_names)
    except (AssertionError, RuntimeError) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print("runtime immutability verified against the bind-mounted baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
