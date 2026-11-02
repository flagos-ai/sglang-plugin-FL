"""MetaX/MACA vendor patches for SGLang internals."""

from __future__ import annotations

import logging
from typing import Callable
from .patches.flashinfer import patch_flashinfer_utils_comm_backend
from .patches.sgl_kernel import patch_sgl_kernel_cutlass_scaled_mm_alias
from .patches.torchvision_image import patch_torchvision_decode_jpeg_cuda_to_cpu
from .patches.sampler import patch_sampler
from .patches.flashinfer_backend import patch_flashinfer_backend_classes
from .patches.fused_moe import patch_fused_moe_functions
from .patches.pynccl_wrapper import patch_pynccl_wrapper
from .patches.prefill_attention import patch_prefill_attention
from .patches.fp32_tp_allreduce import patch_fp32_tp_allreduce
from .patches.triton_attention import patch_triton_attention
from .patches.qwen_vl_image_resize import patch_qwen_vl_image_resize

logger = logging.getLogger(__name__)
_patches_applied = False


# Patches that only make sense when the vendor package they modify is installed.
# In Empty mode (no flashinfer / sgl_kernel) they are skipped instead of failing,
# and must not create shim modules that make the packages look importable.
_PATCH_REQUIRES = {
    "flashinfer_utils_comm_backend": "flashinfer",
    "flashinfer_backend_classes": "flashinfer",
    "sgl_kernel_cutlass_scaled_mm_alias": "sgl_kernel",
}


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _run_required_patch(name: str, patch_func: Callable[[], None]) -> None:
    try:
        patch_func()
    except Exception as error:
        raise RuntimeError(f"MetaX patch failed: {name}") from error


def _required_patches() -> tuple[tuple[str, Callable[[], None]], ...]:
    return (
        (
            "flashinfer_utils_comm_backend",
            patch_flashinfer_utils_comm_backend,
        ),
        (
            "sgl_kernel_cutlass_scaled_mm_alias",
            patch_sgl_kernel_cutlass_scaled_mm_alias,
        ),
        (
            "torchvision_decode_jpeg_cuda_to_cpu",
            patch_torchvision_decode_jpeg_cuda_to_cpu,
        ),
        (
            "flashinfer_backend_classes",
            patch_flashinfer_backend_classes,
        ),
        ("sampler", patch_sampler),
        ("pynccl_wrapper", patch_pynccl_wrapper),
        (
            "fused_moe",
            patch_fused_moe_functions,
        ),
        (
            "prefill_attention",
            patch_prefill_attention,
        ),
        (
            "fp32_tp_allreduce",
            patch_fp32_tp_allreduce,
        ),
        (
            "triton_attention",
            patch_triton_attention,
        ),
        (
            "qwen_vl_image_resize",
            patch_qwen_vl_image_resize,
        ),
    )


def _apply_required_patches() -> None:
    applied, skipped = [], []
    for name, patch_func in _required_patches():
        dep = _PATCH_REQUIRES.get(name)
        if dep is not None and not _has_module(dep):
            skipped.append(f"{name}(no {dep})")
            continue
        _run_required_patch(name, patch_func)
        applied.append(name)

    logger.info("applied MetaX patches: %s", ", ".join(applied))
    if skipped:
        logger.info("skipped MetaX patches: %s", ", ".join(skipped))


def apply_metax_patches() -> None:
    """Apply all MetaX-specific patches."""
    global _patches_applied
    if _patches_applied:
        return

    _apply_required_patches()
    _patches_applied = True


apply_metax_patches()
