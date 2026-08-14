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
logger = logging.getLogger(__name__)
_patches_applied = False


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
        (
            "sampler",
            patch_sampler

        ),
        (
            "pynccl_wrapper",
            patch_pynccl_wrapper
        ),
        (
            "fused_moe",
            patch_fused_moe_functions,
        ),
    )


def _apply_required_patches() -> None:
    for name, patch_func in _required_patches():
        _run_required_patch(name, patch_func)

    logger.info(
        "applied MetaX patches: %s",
        ", ".join(name for name, _ in _required_patches()),
    )


def apply_metax_patches() -> None:
    """Apply all MetaX-specific patches."""
    global _patches_applied
    if _patches_applied:
        return

    _apply_required_patches()
    _patches_applied = True


apply_metax_patches()
