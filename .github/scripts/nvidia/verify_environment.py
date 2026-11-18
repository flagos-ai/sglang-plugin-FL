#!/usr/bin/env python3
"""Verify the NVIDIA operations-runtime SGLang 0.5.18 CI environment.

Two explicit phases, selected by CLI flag:

``--phase pre-setup`` (check.sh, before setup.sh touches anything):
  - NVIDIA GPU visible with the minimum card count
  - Runtime identity: python 3.12, torch 2.13.0+cu130, CUDA 13.0
  - the two image distributions exist (sglang, sglang-fl)
  - no requirement on the active Triton slot, pytest, or any setup.sh
    side effect; imports nothing that mutates the process

``--phase post-setup`` (setup.sh, after the Runtime slot selection):
  - everything pre-setup checks
  - NO standalone triton distribution; ``import triton`` resolves to the
    Runtime's /opt/flagtree module tree (Triton 3.6.0, provided by
    FlagTree). Torch/xgrammar/humming-kernels still declare triton in
    their distribution metadata -- the documented platform capability
    exception (pip check compares exactly against the runtime baseline
    plus those requirers).
  - sglang-plugin-FL 0.1.0 loading from site-packages with both SGLang
    entry-point groups discoverable
  - the ephemeral CI test-runner versions and USE_FLAGGEMS=1
  - with --require-gpu: the plugin's CUDA compatibility view is active and
    FlashInfer's JIT resolves CUDA_HOME to it
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

PHASE_PRE_SETUP = "pre-setup"
PHASE_POST_SETUP = "post-setup"

# Operations runtime (flagos-runtime-nvidia-cuda13.3, digest-pinned in the
# containerfile) plus the two staged wheels. The runtime contributes
# torch/flagtree/flag-gems/numpy; the SGLang wheel closure contributes
# sglang/sglang-kernel/flashinfer (triton is intentionally excluded from
# resolution -- the /opt/flagtree module tree is the compiler); the plugin
# wheel is sglang-fl.
EXPECTED_DISTRIBUTIONS = {
    "sglang": "0.5.18",
    "sglang-fl": "0.1.0",
    "torch": "2.13.0+cu130",
    "flag-gems": "5.4.0",
}

# Only meaningful after setup.sh's ephemeral install; never checked in the
# pre-setup phase. Triton is deliberately absent -- see check_triton_slot.
POST_SETUP_DISTRIBUTIONS = {
    "pytest": "9.1.1",
    "pytest-timeout": "2.4.0",
}

# The Runtime's default compiler slot selected by its login-shell profile.
FLAGTREE_MODULE_ROOT = "/opt/flagtree/triton"
EXPECTED_FLAGTREE_VERSION = "3.6.0"

PLUGIN_ENTRY_POINT_GROUPS = ("sglang.srt.platforms", "sglang.srt.plugins")


def _require_distribution(name: str, expected: str) -> None:
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is missing: {name}") from exc
    if actual != expected:
        raise RuntimeError(f"{name} must be {expected}, found {actual}")
    print(f"[cuda-env] {name}={actual}")


def check_identity_and_gpu(min_gpus: int) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Python must be 3.12, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    print(f"[cuda-env] python={sys.version.split()[0]}")

    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        _require_distribution(name, expected)

    torch = importlib.import_module("torch")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"PyTorch must use CUDA 13.0, found {torch.version.cuda}")
    print(f"[cuda-env] torch-cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device_count = torch.cuda.device_count()
    if device_count < min_gpus:
        raise RuntimeError(
            f"at least {min_gpus} CUDA devices are required, found {device_count}"
        )
    devices = ", ".join(
        f"{index}:{torch.cuda.get_device_name(index)}" for index in range(device_count)
    )
    print(f"[cuda-env] visible-gpus={device_count} ({devices})")


def check_triton_slot() -> None:
    # The compiler is a Runtime module tree, not a pip distribution: the
    # standalone triton distribution must be absent while ``import triton``
    # resolves to /opt/flagtree (3.6.0, provided by FlagTree). The /opt/triton
    # 3.7.1 tree stays untouched and unselected (its integrity is the image
    # verifier's compiler-slots gate, not this runtime check).
    try:
        metadata.version("triton")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "a standalone triton distribution is installed; the image must "
            "resolve without one (the compiler is the /opt/flagtree module tree)"
        )
    print(
        "[cuda-env] no standalone triton distribution (platform capability exception)"
    )

    triton_spec = importlib.util.find_spec("triton")
    if (
        triton_spec is None
        or not triton_spec.origin
        or not triton_spec.origin.startswith(FLAGTREE_MODULE_ROOT)
    ):
        found = triton_spec.origin if triton_spec else None
        raise RuntimeError(
            f"triton module must resolve under {FLAGTREE_MODULE_ROOT}, found {found}"
        )
    triton = importlib.import_module("triton")
    if triton.__version__ != EXPECTED_FLAGTREE_VERSION:
        raise RuntimeError(
            f"/opt/flagtree triton must be {EXPECTED_FLAGTREE_VERSION}, "
            f"found {triton.__version__}"
        )
    print(f"[cuda-env] triton-module authoritative at {triton_spec.origin}")
    print(f"[cuda-env] triton={triton.__version__} (FlagTree module tree)")


def check_flaggems_enabled() -> None:
    if os.environ.get("USE_FLAGGEMS") != "1":
        raise RuntimeError(
            f"USE_FLAGGEMS must be 1 (FlagGems dispatch enabled), found "
            f"{os.environ.get('USE_FLAGGEMS')!r}"
        )
    print("[cuda-env] FlagGems enabled (USE_FLAGGEMS=1)")


def check_installed_plugin() -> None:
    plugin = importlib.import_module("sglang_fl")
    plugin_file = Path(plugin.__file__).resolve()
    if "site-packages" not in plugin_file.parts:
        raise RuntimeError(
            f"sglang_fl must come from the installed wheel, found {plugin_file}"
        )
    print(f"[cuda-env] installed plugin at {plugin_file}")

    selected = metadata.entry_points()
    for group in PLUGIN_ENTRY_POINT_GROUPS:
        entries = [
            ep.value for ep in selected.select(group=group) if ep.name == "sglang_fl"
        ]
        if not entries:
            raise RuntimeError(f"entry-point group {group!r} has no 'sglang_fl' entry")
        print(f"[cuda-env] entry-point {group} -> {entries[0]}")


def check_cuda_view() -> None:
    cuda_home = Path(os.environ.get("CUDA_HOME", ""))
    required_view_paths = (
        cuda_home / "bin" / "nvcc",
        cuda_home / "include" / "cuda_runtime.h",
        cuda_home / "lib64" / "libcudart.so",
    )
    if not str(cuda_home) or not all(path.exists() for path in required_view_paths):
        raise RuntimeError(
            "plugin did not activate a complete CUDA compatibility view: "
            f"CUDA_HOME={cuda_home}, paths={required_view_paths}"
        )

    from flashinfer.jit.cpp_ext import get_cuda_path

    get_cuda_path.cache_clear()
    flashinfer_cuda_home = Path(get_cuda_path()).resolve()
    if flashinfer_cuda_home != cuda_home.resolve():
        raise RuntimeError(
            f"FlashInfer JIT did not consume the plugin CUDA view: "
            f"expected {cuda_home}, found {flashinfer_cuda_home}"
        )
    print(f"[cuda-env] FlashInfer JIT CUDA view={flashinfer_cuda_home}")


def print_provenance_record() -> None:
    """One-time provenance echo; a setup probe, not proof that every later
    functional/e2e process uses the wheel -- that is the installed-wheel
    test mode's job."""
    import triton

    sglang = importlib.import_module("sglang")
    print(
        f"[cuda-env] provenance sglang_fl={metadata.version('sglang-fl')} "
        f"at {importlib.import_module('sglang_fl').__file__}"
    )
    print(
        f"[cuda-env] provenance sglang={metadata.version('sglang')} at {sglang.__file__}"
    )
    print(
        f"[cuda-env] provenance triton={triton.__version__} (module tree, "
        f"no distribution) at {triton.__file__}"
    )


def verify(*, phase: str, min_gpus: int = 1) -> None:
    check_identity_and_gpu(min_gpus)

    if phase == PHASE_POST_SETUP:
        for name, expected in POST_SETUP_DISTRIBUTIONS.items():
            _require_distribution(name, expected)
        check_triton_slot()
        check_flaggems_enabled()
        check_installed_plugin()
        print_provenance_record()
        check_cuda_view()
    print(f"[cuda-env] {phase} environment verification passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(PHASE_PRE_SETUP, PHASE_POST_SETUP),
        required=True,
        help="which verification stage to run",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="minimum visible GPU count",
    )
    args = parser.parse_args()
    if args.min_gpus < 1:
        parser.error("--min-gpus must be at least 1")
    verify(phase=args.phase, min_gpus=args.min_gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
