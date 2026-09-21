#!/usr/bin/env python3
"""Verify the pinned Ascend SGLang 0.5.18 empty runtime."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import os
import re
import site
import sys
from importlib import metadata
from pathlib import Path


EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "sglang": "0.5.18",
    "sglang-fl": "0.1.0",
    "torch": "2.8.0",
    "torch-npu": "2.8.0.post2",
    "transformers": "5.12.1",
    "triton": "3.5.0",
    "triton-ascend": "3.2.0",
    "flag-gems": "5.3.0",
    "xgrammar": "0.2.1",
    "compressed-tensors": "0.15.0",
    "sgl-kernel-npu": "2026.5.1",
    "attentions": "0.2",
    "torch-memory-saver": "0.0.8",
}

EXPECTED_CI_DISTRIBUTIONS = {
    "pytest": "9.1.1",
    "pytest-timeout": "2.4.0",
}

EXPECTED_SOURCE_MARKERS = {
    Path("/opt/sglang-0.5.18/.flagos-source-commit"): (
        "sglang",
        "71de97b264b04dcd514cf904003028aefe9775c8",
    ),
    Path("/opt/FlagGems/.flagos-source-commit"): (
        "FlagGems",
        "98fae44cdf2898f39c7f24f080d7c88b83d7c593",
    ),
    Path("/opt/FlagCX/.flagos-source-commit"): (
        "FlagCX",
        "68f069fe4ff2af9e8017b74aee8dee60c59e3b1d",
    ),
}

# These imports are used by the v0.5.18 NPU backend for the model families in
# tests/platforms/ascend.yaml. Check the wheel surface during image build and
# import every module on a real NPU before CI starts. Missing future-model
# kernels must fail explicitly instead of being hidden by an import stub.
REQUIRED_NPU_KERNEL_MODULES = (
    "sgl_kernel_npu.mem_cache.allocator",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.fla.chunk",
    "sgl_kernel_npu.fla.fused_gdn_gating",
    "sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent",
    "sgl_kernel_npu.fla.utils",
    "sgl_kernel_npu.mamba.mamba_state_update_triton",
    "sgl_kernel_npu.mamba.causal_conv1d",
    "sgl_kernel_npu.norm.l1_norm",
    "sgl_kernel_npu.norm.add_rmsnorm_bias",
    "sgl_kernel_npu.norm.split_qkv_rmsnorm_rope",
    "sgl_kernel_npu.activation.swiglu_oai",
    "sgl_kernel_npu.sample.verify_tree_greedy",
)

CANN_VERSION = "8.5.0"
SGLANG_SOURCE_ROOT = Path("/opt/sglang-0.5.18/python/sglang")
STALE_SGLANG_ROOT = Path("/sgl-workspace/sglang")


def _require_distribution(name: str, expected: str) -> None:
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is missing: {name}") from exc
    if actual != expected:
        raise RuntimeError(f"{name} must be {expected}, found {actual}")
    print(f"[ascend-env] {name}={actual}")


def _require_source_markers() -> None:
    for path, (name, expected) in EXPECTED_SOURCE_MARKERS.items():
        try:
            actual = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"{name} source marker is missing: {path}") from exc
        if actual != expected:
            raise RuntimeError(f"{name} source must be {expected}, found {actual}")
        print(f"[ascend-env] {name}-commit={actual}")


def _read_cann_version() -> tuple[Path, str]:
    home = Path(
        os.environ.get("ASCEND_TOOLKIT_HOME")
        or os.environ.get("ASCEND_INSTALL_PATH")
        or "/usr/local/Ascend/ascend-toolkit/latest"
    )
    candidates = (
        home / "version.cfg",
        home / "aarch64-linux" / "ascend_toolkit_install.info",
    )
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?m)^version=([^\s]+)", text)
        if match:
            return path, match.group(1)
        bracketed = re.findall(r"\[([^\]]+)\]", text)
        if bracketed:
            return path, bracketed[-1]
    raise RuntimeError(
        "cannot determine CANN version; checked " + ", ".join(map(str, candidates))
    )


def _require_cann() -> None:
    path, actual = _read_cann_version()
    if not actual.startswith(CANN_VERSION):
        raise RuntimeError(f"CANN must be {CANN_VERSION}, found {actual} in {path}")
    print(f"[ascend-env] CANN={actual} ({path})")


def _require_npu_kernel_surface() -> None:
    files = metadata.files("sgl-kernel-npu") or []
    normalized = {str(path).replace("\\", "/") for path in files}
    missing: list[str] = []
    for module_name in REQUIRED_NPU_KERNEL_MODULES:
        stem = module_name.replace(".", "/")
        if not any(
            path == f"{stem}.py"
            or path.startswith(f"{stem}.")
            or path == f"{stem}/__init__.py"
            for path in normalized
        ):
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "sgl-kernel-npu is missing required v0.5.18 modules: " + ", ".join(missing)
        )
    print(
        "[ascend-env] sgl-kernel-npu module surface OK "
        f"({len(REQUIRED_NPU_KERNEL_MODULES)} modules)"
    )


def _require_flagcx_layout() -> Path:
    root = Path(os.environ.get("FLAGCX_PATH", "/opt/FlagCX"))
    library = root / "build" / "lib" / "libflagcx.so"
    wrapper = root / "plugin" / "interservice" / "flagcx_wrapper.py"
    for path in (library, wrapper):
        if not path.is_file():
            raise RuntimeError(f"required FlagCX artifact is missing: {path}")
    print(f"[ascend-env] FlagCX artifacts={root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    _require_import("plugin.interservice.flagcx_wrapper")
    return library


def _require_import(name: str) -> object:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(
            f"failed to import {name}: {type(exc).__name__}: {exc}"
        ) from exc
    print(f"[ascend-env] imported {name}")
    return module


def _require_plugin(plugin_root: str | None) -> None:
    if importlib.util.find_spec("sglang_fl") is None:
        raise RuntimeError("sglang_fl is not importable")
    if plugin_root is None:
        print("[ascend-env] sglang_fl import spec found")
        return
    module = _require_import("sglang_fl")
    actual = Path(module.__file__).resolve().parent
    expected = Path(plugin_root).resolve() / "sglang_fl"
    if actual != expected:
        raise RuntimeError(f"sglang_fl must load from {expected}, found {actual}")
    print(f"[ascend-env] plugin-source={actual}")


def _require_sglang_import() -> None:
    module = _require_import("sglang")
    module_path = Path(module.__file__).resolve()
    if module_path == STALE_SGLANG_ROOT or STALE_SGLANG_ROOT in module_path.parents:
        raise RuntimeError(
            f"sglang imported from the stale empty-image checkout: {module_path}"
        )

    from_new_source = (
        module_path == SGLANG_SOURCE_ROOT or SGLANG_SOURCE_ROOT in module_path.parents
    )
    from_site_packages = any(
        parent.name in {"site-packages", "dist-packages"}
        for parent in module_path.parents
    )
    if not from_new_source and not from_site_packages:
        raise RuntimeError(
            "sglang must import from the v0.5.18 source or its installed wheel, "
            f"found {module_path}"
        )

    site_roots = {Path(path) for path in site.getsitepackages()}
    user_site = site.getusersitepackages()
    if user_site:
        site_roots.add(Path(user_site))
    stale_pth: list[Path] = []
    for site_root in site_roots:
        for pth_file in site_root.glob("*.pth"):
            try:
                pth_text = pth_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            normalized_text = pth_text.replace("\\", "/")
            stale_editable_name = re.match(
                r"^__editable__\.sglang-\d", pth_file.name.lower()
            )
            stale_editable_finder = re.search(
                r"__editable___sglang_\d", normalized_text.lower()
            )
            if (
                "/sgl-workspace/sglang/python" in normalized_text
                or stale_editable_name
                or stale_editable_finder
            ):
                stale_pth.append(pth_file)
    if stale_pth:
        raise RuntimeError(
            "stale SGLang editable .pth files remain after upgrade: "
            + ", ".join(map(str, stale_pth))
        )
    print(f"[ascend-env] sglang-source={module_path}")


def _require_npu(min_npus: int, flagcx_library: Path) -> None:
    torch = _require_import("torch")
    _require_import("torch_npu")
    _require_import("torchair")
    _require_import("torchair.configs.compiler_config")
    _require_import(
        "torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce"
    )
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

    for module_name in REQUIRED_NPU_KERNEL_MODULES:
        _require_import(module_name)
    flag_gems = _require_import("flag_gems")
    flag_gems_path = Path(flag_gems.__file__).resolve()
    flag_gems_root = Path("/opt/FlagGems")
    if (
        flag_gems_path != flag_gems_root
        and flag_gems_root not in flag_gems_path.parents
    ):
        raise RuntimeError(
            f"flag_gems must import from {flag_gems_root}, found {flag_gems_path}"
        )
    print(f"[ascend-env] FlagGems-source={flag_gems_path}")
    _require_plugin(None)
    _require_import("sglang.srt.server_args")

    from sglang.srt.platforms import current_platform

    if not current_platform.is_npu():
        raise RuntimeError(
            f"active SGLang platform does not report NPU: {type(current_platform)}"
        )
    dispatch_key = current_platform.get_dispatch_key_name()
    if dispatch_key != "npu":
        raise RuntimeError(
            f"Ascend fused-op dispatch key must be npu, got {dispatch_key}"
        )
    allocator_cls = current_platform.get_paged_allocator_cls()
    expected_allocator = (
        "sglang_fl.dispatch.backends.vendor.ascend.allocator."
        "AscendPagedTokenToKVPoolAllocator"
    )
    actual_allocator = f"{allocator_cls.__module__}.{allocator_cls.__name__}"
    if actual_allocator != expected_allocator:
        raise RuntimeError(
            f"Ascend paged allocator must be {expected_allocator}, got {actual_allocator}"
        )
    compile_backend = current_platform.get_compile_backend("npugraph_ex")
    if compile_backend is None:
        raise RuntimeError("Ascend npugraph_ex compiler backend resolved to None")
    print(
        "[ascend-env] platform="
        f"{type(current_platform).__module__}.{type(current_platform).__name__} "
        f"dispatch={dispatch_key} allocator={actual_allocator} "
        f"compiler={type(compile_backend).__name__}"
    )

    try:
        ctypes.CDLL(str(flagcx_library), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(f"failed to load {flagcx_library}: {exc}") from exc
    print(f"[ascend-env] loaded {flagcx_library}")


def verify(
    *,
    require_ci: bool = False,
    require_npu: bool = False,
    min_npus: int = 1,
    plugin_root: str | None = None,
) -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Python must be 3.11, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    print(f"[ascend-env] python={sys.version.split()[0]}")

    for name, expected in EXPECTED_RUNTIME_DISTRIBUTIONS.items():
        _require_distribution(name, expected)
    if require_ci:
        for name, expected in EXPECTED_CI_DISTRIBUTIONS.items():
            _require_distribution(name, expected)

    _require_source_markers()
    _require_cann()
    _require_npu_kernel_surface()
    flagcx_library = _require_flagcx_layout()

    _require_sglang_import()
    # This import is deliberately safe on driver-less image builders. The full
    # server_args import activates the platform plugin and is checked only after
    # torch_npu has proved that real devices are available.
    _require_import("sglang.srt.platforms.interface")
    _require_plugin(plugin_root)

    if require_npu:
        _require_npu(min_npus, flagcx_library)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ci",
        action="store_true",
        help="also require the pinned CI test packages",
    )
    parser.add_argument(
        "--require-npu",
        action="store_true",
        help="also require working torch_npu devices and native NPU modules",
    )
    parser.add_argument(
        "--min-npus",
        type=int,
        default=1,
        help="minimum visible NPU count when --require-npu is enabled",
    )
    parser.add_argument(
        "--plugin-root",
        default=None,
        help="require sglang_fl to import from this checkout root",
    )
    args = parser.parse_args()
    if args.min_npus < 1:
        parser.error("--min-npus must be at least 1")
    verify(
        require_ci=args.require_ci,
        require_npu=args.require_npu,
        min_npus=args.min_npus,
        plugin_root=args.plugin_root,
    )
    print("[ascend-env] environment verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
