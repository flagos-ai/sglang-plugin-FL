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
     baseline; a transitional exact (xgrammar, triton) exception remains until
     the external-index removal gates can run, while the wheel contract below
     independently rejects metadata that could emit it
  4. compiler slots      -- /opt/flagtree carries the recorded Triton, no
     standalone triton distribution exists, and the triton module
     resolves from /opt/flagtree in THIS process
  5. two-wheel contract  -- sglang and sglang_fl distributions, versions,
     entry points, and the plugin resolving from its installed
     distribution root
  6. HCU compat markers  -- the Runtime clang path and the plugin-owned
     early import guard with the scalar_type exemption (the derived SGLang
     wheel's SHA256 is checked as an OCI label by verify_image.sh, not
     in-image)

It also prints the additions manifest as an audit trail; it is output,
never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pip_check_exceptions import pip_check_violations  # noqa: E402

EXPECTED_SGLANG = "0.5.18+flagos.hcu1"
EXPECTED_PLUGIN = "0.1.0"
EXPECTED_XGRAMMAR = "0.2.1+flagos.hcu1"
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}
REQUIRED_PLUGIN_PAYLOAD = (
    "sglang_fl/dispatch/backends/vendor/hygon/early.py",
    "sglang_fl/dispatch/backends/vendor/hygon/import_guard.py",
    "sglang_fl/dispatch/backends/vendor/hygon/impl/attention_backend.py",
    "sglang_fl/dispatch/config/hygon.yaml",
)
RUNTIME_CLANG = Path("/opt/dtk/aillvm/bin/clang-18")


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
    print("[3/6] pip check: no violation beyond the transitional exact exception")


def check_compiler_slots(evidence: Path) -> None:
    recorded = json.loads((evidence / "runtime-0.5.18.provenance.json").read_text())
    slots = recorded.get("compiler_slots", {})
    assert slots, "provenance.json carries no compiler slots"
    # The schema keys are slot ROOTS ('/opt/flagtree', '/opt/triton');
    # every recorded slot must still carry its recorded Triton version.
    for slot_root, expected_version in sorted(slots.items()):
        init_path = Path(slot_root) / "triton" / "__init__.py"
        assert init_path.is_file(), f"compiler slot missing: {init_path}"
        assert expected_version, f"provenance.json lacks the version for {slot_root}"
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", init_path.read_text())
        actual = match.group(1) if match else None
        assert actual == expected_version, (
            f"compiler slot {slot_root} drifted: expected {expected_version}, "
            f"found {actual}"
        )
        print(f"[4/6] compiler slot {slot_root}: triton {actual}")
    current = installed_distributions()
    assert "triton" not in current, (
        f"standalone triton distribution present: {current['triton']}"
    )
    selected = recorded.get("selected_slot")
    assert selected in slots, f"selected slot {selected!r} is not a recorded slot"
    # find_spec resolves without executing the module: safe on GPU-less
    # builders, and it is the resolution every entrypoint inherits from
    # the image-level PYTHONPATH.
    spec = importlib.util.find_spec("triton")
    assert spec is not None and spec.origin, "triton module not resolvable"
    assert spec.origin.startswith(str(Path(selected).resolve()) + "/"), (
        f"triton module resolves {spec.origin}, expected under {selected}"
    )
    print(
        f"[4/6] no standalone triton; module resolves {spec.origin} (slot {selected})"
    )


def check_two_wheel_contract() -> None:
    """Metadata/payload-only: this verifier must run on GPU-less builders.

    Importing sglang_fl (or the attention backend) initializes HCU and
    aborts with 'no ROCm-capable device' without a GPU, so the wheel
    contract is checked through distribution metadata, file listings and
    payload paths; the real import runs on the HCU CI runner.
    """
    import importlib.metadata as md

    assert md.version("sglang") == EXPECTED_SGLANG, md.version("sglang")
    assert md.version("sglang-fl") == EXPECTED_PLUGIN, md.version("sglang-fl")
    assert md.version("xgrammar") == EXPECTED_XGRAMMAR, md.version("xgrammar")
    xgrammar = md.distribution("xgrammar")
    triton_requirements = [
        requirement
        for requirement in xgrammar.requires or ()
        if canonicalize_name(Requirement(requirement).name) == "triton"
    ]
    assert not triton_requirements, (
        f"corrected xgrammar metadata still declares triton: {triton_requirements}"
    )
    for group, (name, value) in EXPECTED_ENTRY_POINTS.items():
        registered = {(ep.name, ep.value) for ep in md.entry_points(group=group)}
        assert (name, value) in registered, (
            f"{group}: ({name!r}, {value!r}) not in {sorted(registered)}"
        )
    dist = md.distribution("sglang-fl")
    distribution_root = Path(dist.locate_file("")).resolve()
    payload = {str(item) for item in dist.files or ()}
    for required in REQUIRED_PLUGIN_PAYLOAD:
        assert required in payload, (
            f"{required} missing from the installed sglang_fl distribution"
        )
        assert (distribution_root / required).is_file(), (
            f"{required} listed in RECORD but absent on disk"
        )
    print(
        f"[5/6] two-wheel contract: sglang=={EXPECTED_SGLANG}, "
        f"sglang_fl=={EXPECTED_PLUGIN} at {distribution_root}; corrected "
        f"xgrammar=={EXPECTED_XGRAMMAR} has no triton requirement "
        f"(metadata/payload verified; import deferred to HCU CI)"
    )


def check_hcu_compat_markers() -> None:
    import importlib.metadata as md

    assert RUNTIME_CLANG.is_file(), f"Runtime clang missing: {RUNTIME_CLANG}"
    assert "TRITON_HIP_CLANG_PATH" not in os.environ, (
        "legacy clang wrapper override remains in the image"
    )
    dist = md.distribution("sglang-fl")
    distribution_root = Path(dist.locate_file("")).resolve()
    guard = (
        distribution_root / "sglang_fl/dispatch/backends/vendor/hygon/import_guard.py"
    )
    text = guard.read_text()
    assert 'fullname == "sgl_kernel.scalar_type"' in text, (
        "plugin import guard lacks the scalar_type exemption"
    )
    purelib = Path(sysconfig.get_paths()["purelib"])
    assert not (purelib / "_hcu_kernel_stub_finder.py").exists(), (
        "legacy image-written stub finder remains"
    )
    assert not (purelib / "zz-hcu-kernel-stub-hook.pth").exists(), (
        "legacy startup hook remains"
    )
    print("[6/6] HCU compat markers: Runtime clang + plugin early import guard")


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
