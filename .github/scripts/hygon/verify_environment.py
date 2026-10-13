#!/usr/bin/env python3
"""Verify the documented Hygon empty runtime after the SGLang 0.5.18 upgrade.

Metadata-only checks work during an ordinary Docker build. Full checks load the
vendor runtime and FlagCX library; --require-gpu additionally executes vendor
RMSNorm to expose torch/sgl-kernel ABI and kernel failures before model loading.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from importlib import metadata
from pathlib import Path


def _default_plugin_root(script_path: Path) -> Path:
    # The Docker baseline stage copies this script directly into /opt.
    # That flat path has no repository ancestors; capture needs none.
    if script_path.parent.parts[-3:] == (".github", "scripts", "hygon"):
        return script_path.parents[3]
    return Path.cwd()


REPO_ROOT = _default_plugin_root(Path(__file__).resolve())
REQUIRED_VENDOR_DISTRIBUTIONS = (
    "torch",
    "sglang-kernel",
    "flagtree",
    "flag-gems",
)
OPTIONAL_VENDOR_DISTRIBUTIONS = (
    "torchvision",
    "torchaudio",
    "triton",
    "flash-attn",
)
EXPECTED_RELEASES = {
    "sglang": "0.5.18",
    "torch": "2.10.0",
    "sglang-kernel": "0.4.2.post2",
    "flag-gems": "5.4.0.dev0",
}
EXPECTED_FLAGTREE = "0.6.1+hcu3.6"


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is missing: {name}") from exc


def capture_vendor_stack(path: Path) -> None:
    """Record the actual vendor metadata, including its DTK local versions."""
    versions = {
        name: _distribution_version(name) for name in REQUIRED_VENDOR_DISTRIBUTIONS
    }
    for name in OPTIONAL_VENDOR_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    path.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    constraints_path = path.with_suffix(".constraints.txt")
    constraints_path.write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items()),
        encoding="utf-8",
    )
    print(f"[hygon-env] vendor baseline: {path}")
    print(f"[hygon-env] vendor constraints: {constraints_path}")


def _verify_versions(baseline: Path | None) -> None:
    for name, expected in EXPECTED_RELEASES.items():
        actual = _distribution_version(name)
        # DTK wheels encode their build and ABI in PEP 440 local versions.
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(f"{name} must have release {expected}, found {actual}")
        print(f"[hygon-env] {name}={actual}")
    actual = _distribution_version("flagtree")
    if actual != EXPECTED_FLAGTREE:
        raise RuntimeError(f"flagtree must be {EXPECTED_FLAGTREE}, found {actual}")
    print(f"[hygon-env] flagtree={actual}")

    if baseline is not None:
        try:
            versions = json.loads(baseline.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"could not read vendor baseline {baseline}: {exc}"
            ) from exc
        if not isinstance(versions, dict) or not versions:
            raise RuntimeError(
                f"vendor baseline must be a non-empty mapping: {baseline}"
            )
        missing = set(REQUIRED_VENDOR_DISTRIBUTIONS) - set(versions)
        if missing:
            raise RuntimeError(
                f"vendor baseline is missing distributions: {sorted(missing)}"
            )
        for name, expected in versions.items():
            actual = _distribution_version(name)
            if actual != expected:
                raise RuntimeError(
                    f"vendor stack changed after the upgrade: {name} was "
                    f"{expected}, now {actual}; restore the base image vendor wheel"
                )
        print("[hygon-env] vendor wheel versions match the captured base image")


def _verify_plugin_path(plugin_root: Path) -> None:
    spec = importlib.util.find_spec("sglang_fl")
    if spec is None or spec.origin is None:
        raise RuntimeError("sglang_fl is not importable; install the current checkout")
    actual = Path(spec.origin).resolve().parent
    expected = plugin_root.resolve() / "sglang_fl"
    if actual != expected:
        raise RuntimeError(
            f"sglang_fl resolves to {actual}, expected {expected}; "
            "an older plugin copy may be shadowing the upgrade"
        )
    print(f"[hygon-env] plugin-source={actual}")


def _require_import(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"runtime import failed: {module_name}: {type(exc).__name__}: {exc}. "
            "Install a compatible Python dependency or restore the matching "
            "Hygon vendor wheel; do not replace DTK torch with a PyPI CUDA wheel."
        ) from exc
    print(f"[hygon-env] imported {module_name}")
    return module


def _verify_hygon_torch(torch) -> None:
    hcu_version = getattr(torch, "__hcu_version__", None)
    hip_version = getattr(torch.version, "hip", None)
    if not hcu_version and not hip_version:
        raise RuntimeError(
            "torch has neither __hcu_version__ nor torch.version.hip; "
            "the Hygon HCU/HIP PyTorch build is required"
        )
    print(f"[hygon-env] torch-hcu={hcu_version}, torch-hip={hip_version}")


def _verify_flash_attention() -> None:
    module = _require_import("flash_attn")
    for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"vendor flash_attn is missing callable {name}")
    print("[hygon-env] vendor FlashAttention prefill/decode APIs are available")


def _verify_flagcx() -> None:
    raw_path = os.environ.get("FLAGCX_PATH", "").strip()
    if not raw_path:
        raise RuntimeError(
            "FLAGCX_PATH must point to the preserved FlagCX installation"
        )
    library_path = Path(raw_path) / "build" / "lib" / "libflagcx.so"
    if not library_path.is_file():
        raise RuntimeError(f"FlagCX shared library is missing: {library_path}")
    module = _require_import("sglang_fl.distributed.device_communicators.flagcx")
    try:
        wrapper_cls = module._import_flagcx_wrapper()[0]
        library = wrapper_cls(str(library_path))
    except Exception as exc:
        raise RuntimeError(
            f"FlagCX wrapper/shared library could not load from {library_path}: {exc}"
        ) from exc
    for name in (
        "flagcxGetUniqueId",
        "flagcxCommInitRank",
        "flagcxAllReduce",
        "flagcxSend",
        "flagcxRecv",
    ):
        if not callable(getattr(library, name, None)):
            raise RuntimeError(f"FlagCX library is missing callable {name}")
    print(f"[hygon-env] FlagCX wrapper and shared library loaded: {library_path}")
    print(
        "[hygon-env] FlagCX collectives still require the TP/multinode acceptance tests"
    )


def _verify_gpu(torch, min_gpus: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false on the Hygon runtime")
    count = torch.cuda.device_count()
    if count < min_gpus:
        raise RuntimeError(
            f"at least {min_gpus} Hygon GPUs are required, found {count}"
        )
    devices = ", ".join(
        f"{index}:{torch.cuda.get_device_name(index)}" for index in range(count)
    )
    print(f"[hygon-env] visible-gpus={count} ({devices})")

    kernel = _require_import("sgl_kernel")
    try:
        x = (torch.arange(128, device="cuda", dtype=torch.float32) / 64).reshape(2, 64)
        x = x.to(torch.bfloat16)
        weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        # Match the documented vendor wheel's out-parameter ABI.
        kernel.rmsnorm(out, x, weight, 1e-6)
        torch.cuda.synchronize()
        expected = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        )
        if not torch.allclose(out.float(), expected, rtol=2e-2, atol=2e-2):
            raise RuntimeError(
                "vendor RMSNorm output does not match the torch reference"
            )
    except Exception as exc:
        raise RuntimeError(
            f"Hygon torch/sgl-kernel RMSNorm check failed: {type(exc).__name__}: {exc}"
        ) from exc
    print("[hygon-env] Hygon torch/sgl-kernel BF16 RMSNorm execution passed")


def verify(
    *,
    plugin_root: Path = REPO_ROOT,
    baseline: Path | None = None,
    metadata_only: bool = False,
    require_gpu: bool = False,
    min_gpus: int = 1,
) -> None:
    if metadata_only and require_gpu:
        raise RuntimeError("--metadata-only cannot be combined with --require-gpu")
    if min_gpus < 1:
        raise RuntimeError("--min-gpus must be at least 1")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"Python must be 3.10 for the documented Hygon wheels, found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    print(f"[hygon-env] python={sys.version.split()[0]}")
    _verify_versions(baseline)
    _verify_plugin_path(plugin_root)
    if metadata_only:
        print("[hygon-env] metadata checks passed; runtime/GPU checks were not run")
        return

    torch = _require_import("torch")
    _verify_hygon_torch(torch)
    triton = _require_import("triton")
    if str(getattr(triton, "__version__", "")).split(".")[:2] != ["3", "6"]:
        raise RuntimeError(
            f"FlagTree must provide Triton 3.6, found {getattr(triton, '__version__', None)}"
        )
    # Import SGLang first to preserve its real plugin-discovery lifecycle.
    for name in (
        "sgl_kernel",
        "sglang",
        "sglang_fl",
        "sglang.srt.server_args",
        "sqlalchemy",
        "xgrammar",
        "compressed_tensors",
    ):
        _require_import(name)
    _verify_flash_attention()
    _require_import("flag_gems")
    _require_import("sglang_fl.dispatch.backends.vendor.hygon.impl.attention_backend")
    _verify_flagcx()
    if require_gpu:
        _verify_gpu(torch, min_gpus)
    print("[hygon-env] runtime imports passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument(
        "--capture-vendor-stack",
        type=Path,
        help="record base-image versions and adjacent .constraints.txt, then exit",
    )
    args = parser.parse_args(argv)
    if args.min_gpus < 1:
        parser.error("--min-gpus must be at least 1")
    if args.metadata_only and args.require_gpu:
        parser.error("--metadata-only cannot be combined with --require-gpu")
    try:
        if args.capture_vendor_stack is not None:
            capture_vendor_stack(args.capture_vendor_stack)
        else:
            verify(
                plugin_root=args.plugin_root,
                baseline=args.baseline,
                metadata_only=args.metadata_only,
                require_gpu=args.require_gpu,
                min_gpus=args.min_gpus,
            )
    except RuntimeError as exc:
        print(f"[hygon-env] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
