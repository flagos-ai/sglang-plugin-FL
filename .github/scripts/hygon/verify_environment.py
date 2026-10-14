#!/usr/bin/env python3
"""Verify the documented Hygon empty runtime after the SGLang 0.5.18 upgrade.

Metadata-only checks work during an ordinary Docker build. Full checks load the
vendor runtime and FlagCX library; --require-gpu additionally compares the
selected HCU BF16 torch, FlagOS SiLU-and-multiply, and reference GemmaRMSNorm
paths against CPU results. The empty image does not require native RMSNorm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace


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


_TRITON_REQUIRED_FILES = ("__init__.py", "compiler/compiler.py")
_TRITON_OPTIONAL_FILES = ("_C/libtriton.so",)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triton_module_snapshot() -> dict:
    # Only look up the top-level package. Looking up triton.compiler.compiler
    # would import the parent package and could load Torch or GPU libraries.
    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        raise RuntimeError("FlagTree-provided triton package could not be located")
    root = Path(spec.origin).resolve().parent
    files = {}
    for relative in _TRITON_REQUIRED_FILES + _TRITON_OPTIONAL_FILES:
        path = root / relative
        if not path.is_file():
            if relative in _TRITON_REQUIRED_FILES:
                raise RuntimeError(
                    f"required FlagTree-provided triton file is missing: {path}"
                )
            continue
        try:
            files[relative] = _file_sha256(path)
        except OSError as exc:
            raise RuntimeError(
                f"could not hash FlagTree-provided triton file {path}: {exc}"
            ) from exc
    return {"schema_version": 1, "triton": {"package_root": str(root), "files": files}}


def _verify_vendor_modules(baseline: Path) -> None:
    modules_path = baseline.with_suffix(".modules.json")
    try:
        captured = json.loads(modules_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"vendor module baseline is missing: {modules_path}; module hashes "
            "were not checked. Re-run --capture-vendor-stack in the original "
            "unmodified base image, not in the upgraded environment."
        ) from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"could not read vendor module baseline {modules_path}: {exc}"
        ) from exc
    if not isinstance(captured, dict) or captured.get("schema_version") != 1:
        raise RuntimeError(f"invalid vendor module baseline schema: {modules_path}")
    triton = captured.get("triton")
    files = triton.get("files") if isinstance(triton, dict) else None
    if not isinstance(files, dict) or not set(_TRITON_REQUIRED_FILES) <= set(files):
        raise RuntimeError(
            f"vendor module baseline is missing required triton file hashes: {modules_path}"
        )
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in files.values()
    ):
        raise RuntimeError(f"invalid SHA256 in vendor module baseline: {modules_path}")
    current = _triton_module_snapshot()["triton"]["files"]
    if set(files) != set(current):
        raise RuntimeError(
            "FlagTree triton module file set changed after the upgrade: "
            f"captured={sorted(files)}, current={sorted(current)}"
        )
    for relative, expected in files.items():
        if current[relative] != expected:
            raise RuntimeError(
                f"FlagTree triton module hash changed after the upgrade: {relative}; "
                "distribution versions can stay unchanged when pip overwrites "
                "shared module files. Restore the preserved FlagTree package."
            )
    print(
        f"[hygon-env] FlagTree triton module SHA256 checks passed ({len(files)} files)"
    )


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
    modules = _triton_module_snapshot()
    path.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    constraints_path = path.with_suffix(".constraints.txt")
    constraints_path.write_text(
        "".join(f"{name}=={version}\n" for name, version in versions.items()),
        encoding="utf-8",
    )
    modules_path = path.with_suffix(".modules.json")
    modules_path.write_text(json.dumps(modules, indent=2) + "\n", encoding="utf-8")
    print(f"[hygon-env] vendor baseline: {path}")
    print(f"[hygon-env] vendor constraints: {constraints_path}")
    print(f"[hygon-env] vendor module SHA256 baseline: {modules_path}")


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
        for name in OPTIONAL_VENDOR_DISTRIBUTIONS:
            if name in versions:
                continue
            try:
                added_version = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            raise RuntimeError(
                f"vendor stack gained distribution absent from the base image: "
                f"{name}=={added_version}; it can overwrite shared vendor modules. "
                "Remove the added distribution and restore the preserved vendor package."
            )
        print("[hygon-env] vendor wheel versions match the captured base image")
        _verify_vendor_modules(baseline)


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


def _report_sgl_kernel(torch, kernel) -> None:
    """Inspect the empty wheel without requiring an unselected native kernel."""
    native_rmsnorm = getattr(torch.ops.sgl_kernel, "rmsnorm", None)
    wrapper = getattr(kernel, "rmsnorm", None)
    try:
        wrapper_signature = (
            str(inspect.signature(wrapper)) if callable(wrapper) else None
        )
    except (TypeError, ValueError):
        wrapper_signature = "unavailable"
    print(
        f"[hygon-env] sgl_kernel={getattr(kernel, '__file__', None)}, "
        f"rmsnorm-wrapper-signature={wrapper_signature}, "
        f"native-rmsnorm-registered={native_rmsnorm is not None} (informational)"
    )
    print(
        "[hygon-env] native RMSNorm ABI/execution was not validated; "
        "rms_norm is not selected in the documented empty-image policy"
    )


def _compare_bf16(torch, actual, expected, *, name: str, device: str) -> None:
    if not isinstance(actual, torch.Tensor):
        raise RuntimeError(f"{name} did not return a tensor")
    if actual.dtype != torch.bfloat16 or actual.device.type != device:
        raise RuntimeError(
            f"{name} returned dtype/device {actual.dtype}/{actual.device}; "
            f"expected bfloat16/{device}"
        )
    try:
        torch.testing.assert_close(
            actual.detach().float().cpu(),
            expected.detach().float().cpu(),
            rtol=2e-2,
            atol=2e-2,
        )
    except AssertionError as exc:
        raise RuntimeError(f"{name} disagrees with its CPU reference: {exc}") from exc


def _verify_selected_tensor_paths(
    torch, *, device: str, silu_and_mul, gemma_rms_norm
) -> None:
    """Numerically check selected paths on the requested actual tensor device.

    CPU regression tests call this with CPU implementations and real CPU tensors.
    Only _verify_gpu may label a successful CUDA-device invocation as GPU-tested.
    """
    x_cpu = (torch.arange(128, dtype=torch.float32).reshape(2, 64) / 64 - 1).to(
        torch.bfloat16
    )
    matrix_cpu = (
        (torch.arange(512, dtype=torch.float32).reshape(64, 8) % 13 - 6) / 16
    ).to(torch.bfloat16)
    x = x_cpu.to(device=device)
    matrix = matrix_cpu.to(device=device)
    _compare_bf16(
        torch,
        torch.mm(x, matrix),
        torch.mm(x_cpu.float(), matrix_cpu.float()),
        name="native torch BF16 matmul",
        device=device,
    )

    activation_cpu = (
        torch.arange(256, dtype=torch.float32).reshape(2, 128) / 64 - 2
    ).to(torch.bfloat16)
    left, right = activation_cpu.float().chunk(2, dim=-1)
    activation_expected = torch.nn.functional.silu(left) * right
    activation_actual = silu_and_mul(None, activation_cpu.to(device=device))
    _compare_bf16(
        torch,
        activation_actual,
        activation_expected,
        name="selected silu_and_mul",
        device=device,
    )

    weight_cpu = torch.linspace(-0.5, 0.5, 64).to(torch.bfloat16)
    obj = SimpleNamespace(weight=weight_cpu.to(device=device), variance_epsilon=1e-6)
    expected = x_cpu.float() / torch.sqrt(
        x_cpu.float().square().mean(-1, keepdim=True) + obj.variance_epsilon
    )
    expected = expected * (1 + weight_cpu.float())
    _compare_bf16(
        torch,
        gemma_rms_norm(obj, x),
        expected,
        name="selected GemmaRMSNorm reference",
        device=device,
    )

    residual_cpu = (x_cpu.flip(-1).float() / 8).to(torch.bfloat16)
    combined_cpu = x_cpu + residual_cpu
    expected = combined_cpu.float() / torch.sqrt(
        combined_cpu.float().square().mean(-1, keepdim=True) + obj.variance_epsilon
    )
    expected = expected * (1 + weight_cpu.float())
    normalized, residual = gemma_rms_norm(obj, x, residual_cpu.to(device=device))
    _compare_bf16(
        torch,
        normalized,
        expected,
        name="selected GemmaRMSNorm reference with residual",
        device=device,
    )
    _compare_bf16(
        torch,
        residual,
        combined_cpu,
        name="selected GemmaRMSNorm residual sum",
        device=device,
    )
    print(
        f"[hygon-env] selected tensor-path numerical checks passed on device={device}"
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
    activation = _require_import("sglang_fl.dispatch.backends.flagos.impl.activation")
    normalization = _require_import(
        "sglang_fl.dispatch.backends.reference.impl.gemma_rms_norm"
    )
    try:
        _verify_selected_tensor_paths(
            torch,
            device="cuda",
            silu_and_mul=activation.silu_and_mul_flagos,
            gemma_rms_norm=normalization.gemma_rms_norm_torch,
        )
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"selected Hygon GPU path check failed: {type(exc).__name__}: {exc}"
        ) from exc
    print(
        "[hygon-env] HCU BF16 torch, FlagOS silu_and_mul, and reference "
        "GemmaRMSNorm execution passed; native RMSNorm was not executed"
    )


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
    _report_sgl_kernel(torch, _require_import("sgl_kernel"))
    # Import SGLang before the plugin to preserve its discovery lifecycle.
    for name in (
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
    else:
        print("[hygon-env] GPU selected-op execution was not run")
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
