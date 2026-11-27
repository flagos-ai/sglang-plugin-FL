#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prove the installed-wheel contract of the Ascend two-wheel CI image.

setup.sh runs this program twice, around the pinned test-runner install:

  --capture-runtime-snapshot FILE   record the distribution inventory
  --verify --runtime-evidence FILE  re-read it and check the contract

Only plugin-owned facts are asserted: the SGLang and sglang_fl wheels the
image installs, the entry points SGLang resolves them by, no standalone
triton distribution, and the plugin's sgl_kernel_npu placeholder with its
``npu::causal_conv1d`` fallback registration. What the Runtime owns
(CANN, torch_npu, FlagTree, FlagGems, FlagCX versions) is printed, never
asserted -- the Runtime is read-only and may move under us.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys

EXPECTED_WHEELS = {"sglang": "0.5.18", "sglang-fl": "0.1.0"}
# Exact (name, value): SGLang loads the plugin through these entry points, so
# a value typo that still parses would silently break activation.
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}
# The plugin registers this schema through the sgl_kernel_npu fallbacks while
# the placeholder is active (vendor/ascend/sgl_kernel_npu_fallbacks.py); the
# GDN backend launches the op by it on every step.
EXPECTED_CAUSAL_CONV1D_SCHEMA_PREFIX = (
    "npu::causal_conv1d(Tensor x, Tensor weight, Tensor(a!) conv_states, "
)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_inventory(exclude_root: Path | None = None) -> dict[str, str]:
    """Canonical distribution name -> version.

    Anything located under ``exclude_root`` (the checkout) is skipped so that
    a source tree can never pass for an installed wheel.
    """
    from importlib import metadata

    inventory = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        if exclude_root is not None:
            origin = Path(dist.locate_file("")).resolve()
            if origin == exclude_root or exclude_root in origin.parents:
                continue
        inventory[canonical(name)] = dist.version
    return inventory


def inventory_drift(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Runtime distributions that changed or disappeared; additions are not drift."""
    return sorted(
        f"{name}: {version} -> {after.get(name, '<REMOVED>')}"
        for name, version in before.items()
        if after.get(name) != version
    )


def entry_point_failures(groups: dict[str, dict[str, str]]) -> list[str]:
    return [
        f"entry point {group}/{name} must be exactly {value!r}, "
        f"found {groups.get(group, {}).get(name)!r}"
        for group, (name, value) in EXPECTED_ENTRY_POINTS.items()
        if groups.get(group, {}).get(name) != value
    ]


def collected_entry_points() -> dict[str, dict[str, str]]:
    from importlib import metadata

    groups: dict[str, dict[str, str]] = {}
    for entry_point in metadata.entry_points():
        if entry_point.group in EXPECTED_ENTRY_POINTS:
            groups.setdefault(entry_point.group, {})[entry_point.name] = (
                entry_point.value
            )
    return groups


def wheel_origin_failures(plugin_root: Path) -> list[str]:
    """The plugin under test is the installed wheel, never the checkout."""
    from importlib import metadata

    import sglang_fl

    root = Path(metadata.distribution("sglang-fl").locate_file("")).resolve()
    module_file = Path(sglang_fl.__file__).resolve()
    if not module_file.is_relative_to(root):
        return [
            f"sglang_fl resolves from {module_file}, outside the installed "
            f"distribution root {root}"
        ]
    if module_file.is_relative_to(plugin_root):
        return [f"sglang_fl resolves from the checkout {plugin_root}: {module_file}"]
    return []


def placeholder_failures() -> list[str]:
    """Without an sgl_kernel_npu wheel the placeholder carries the GDN kernels."""
    from importlib import metadata

    try:
        version = metadata.version("sgl-kernel-npu")
    except metadata.PackageNotFoundError:
        pass
    else:
        print(
            f"[ascend-env] sgl_kernel_npu: real distribution {version}, "
            "placeholder checks skipped"
        )
        return []

    # Import the early hook the engine's platform activation runs: it both
    # installs the placeholder and registers the causal-conv1d fallback, so
    # the checks below see exactly the contract the engine gets.
    from sglang_fl.dispatch.backends.vendor.ascend import early  # noqa: F401
    from sglang_fl.dispatch.backends.vendor.ascend import import_guard

    if "sgl_kernel_npu" not in import_guard.install_import_guards():
        return ["the sgl_kernel_npu placeholder is not active"]

    failures: list[str] = []
    # The import statement is the path SGLang takes; a plain attribute access
    # would only reach the generic call-fails-loudly placeholder.
    import sgl_kernel_npu.fla.chunk as chunk
    import sgl_kernel_npu.norm.l1_norm as l1_norm_module

    if not callable(l1_norm_module.l1_norm):
        failures.append("sgl_kernel_npu.norm.l1_norm.l1_norm must carry the fallback")
    if not callable(chunk.chunk_gated_delta_rule_npu):
        failures.append(
            "sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu must carry the fallback"
        )
    # SGLang selects its rope fast path through try/except ImportError; the
    # symbol must stay absent so the generic path takes over.
    try:
        from sgl_kernel_npu.norm import fused_rope_qk_mqa  # noqa: F401
    except ImportError:
        pass
    else:
        failures.append("fused_rope_qk_mqa must stay absent for SGLang's fallback")
    import torch

    try:
        schema = str(torch.ops.npu.causal_conv1d.default._schema)
    except (AttributeError, RuntimeError) as exc:
        failures.append(f"npu::causal_conv1d is not registered: {exc}")
    else:
        if not schema.startswith(EXPECTED_CAUSAL_CONV1D_SCHEMA_PREFIX):
            failures.append("npu::causal_conv1d schema mismatch: " + schema)
    return failures


def _print_runtime_facts() -> None:
    """Print what the Runtime owns; never assert it (the Runtime is read-only)."""
    from importlib import metadata

    for name in ("torch", "torch-npu", "flag-gems", "flagtree", "flagcx"):
        try:
            print(f"[ascend-env] runtime {name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"[ascend-env] runtime {name}=<no distribution>")
    spec = importlib.util.find_spec("triton")
    if spec is not None and spec.origin:
        print(f"[ascend-env] runtime triton module={spec.origin}")
    for path in (
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path(
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/"
            "ascend_all_cann_install.info"
        ),
    ):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?m)^version=([^\s]+)", text)
        if match:
            print(f"[ascend-env] runtime CANN={match.group(1)} ({path})")
            break


def _require_npu(min_npus: int) -> None:
    import torch

    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    count = torch.npu.device_count()
    if count < min_npus:
        raise RuntimeError(f"at least {min_npus} NPUs are required, found {count}")
    devices = ", ".join(
        f"{index}:{torch.npu.get_device_name(index)}" for index in range(count)
    )
    print(f"[ascend-env] visible-npus={count} ({devices})")

    torch.npu.set_device(0)
    probe = torch.arange(4, dtype=torch.float32, device="npu") + 1
    torch.npu.synchronize()
    if probe.cpu().tolist() != [1.0, 2.0, 3.0, 4.0]:
        raise RuntimeError(f"unexpected NPU probe result: {probe.cpu().tolist()}")
    print("[ascend-env] NPU tensor probe passed")

    from sglang.srt.plugins import load_plugins

    load_plugins()
    import sglang_fl

    if not sglang_fl.is_plugin_active():
        raise RuntimeError("sglang_fl general plugin did not become active")
    print("[ascend-env] sglang_fl general plugin active")

    from sglang.srt.platforms import current_platform

    if not current_platform.is_npu():
        raise RuntimeError(
            f"active SGLang platform does not report NPU: {type(current_platform)}"
        )
    print(
        "[ascend-env] platform="
        f"{type(current_platform).__module__}.{type(current_platform).__name__}"
    )


def verify(
    *,
    runtime_evidence: Path | None = None,
    require_npu: bool = False,
    min_npus: int = 1,
    plugin_root: Path,
) -> int:
    plugin_root = plugin_root.resolve()
    after = runtime_inventory(plugin_root)
    before = json.loads(runtime_evidence.read_text()) if runtime_evidence else {}

    failures: list[str] = []
    if runtime_evidence is not None:
        failures += inventory_drift(before, after)
    failures += [
        f"{name} must be {version}, found {after.get(name)}"
        for name, version in EXPECTED_WHEELS.items()
        if after.get(name) != version
    ]
    failures += entry_point_failures(collected_entry_points())
    failures += wheel_origin_failures(plugin_root)
    failures += placeholder_failures()
    if "triton" in after:
        failures.append(
            f"a standalone triton distribution is installed: {after['triton']}"
        )

    _print_runtime_facts()
    if runtime_evidence is not None:
        print(f"[ascend-env] test tooling added: {sorted(set(after) - set(before))}")
    if require_npu:
        _require_npu(min_npus)

    for failure in failures:
        print(f"::error:: {failure}", file=sys.stderr)
    if not failures:
        print("[ascend-env] environment verification passed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-runtime-snapshot", type=Path, metavar="FILE")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--runtime-evidence", type=Path, metavar="FILE")
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument("--min-npus", type=int, default=1)
    parser.add_argument("--plugin-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.min_npus < 1:
        parser.error("--min-npus must be at least 1")
    if args.capture_runtime_snapshot is not None:
        inventory = runtime_inventory(args.plugin_root.resolve())
        args.capture_runtime_snapshot.write_text(json.dumps(inventory, sort_keys=True))
        print(f"[ascend-env] runtime snapshot: {len(inventory)} distributions")
        return 0
    if not args.capture_runtime_snapshot and not args.verify:
        parser.error("--capture-runtime-snapshot or --verify")
    # --verify without evidence is the device-preflight form check.sh uses;
    # drift checking only applies when setup.sh supplies the snapshot.
    return verify(
        runtime_evidence=args.runtime_evidence,
        require_npu=args.require_npu,
        min_npus=args.min_npus,
        plugin_root=args.plugin_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
