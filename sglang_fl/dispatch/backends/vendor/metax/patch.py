"""MetaX/MACA vendor patches for SGLang internals."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)
_patches_applied = False

# (name, submodule under .patches, function). Submodules are imported lazily so a
# patch that does not apply to the installed SGLang / vendor stack cannot break
# the import of the others.
_PATCHES = (
    (
        "flashinfer_utils_comm_backend",
        "flashinfer",
        "patch_flashinfer_utils_comm_backend",
    ),
    (
        "sgl_kernel_cutlass_scaled_mm_alias",
        "sgl_kernel",
        "patch_sgl_kernel_cutlass_scaled_mm_alias",
    ),
    (
        "torchvision_decode_jpeg_cuda_to_cpu",
        "torchvision_image",
        "patch_torchvision_decode_jpeg_cuda_to_cpu",
    ),
    (
        "flashinfer_backend_classes",
        "flashinfer_backend",
        "patch_flashinfer_backend_classes",
    ),
    ("sampler", "sampler", "patch_sampler"),
    ("pynccl_wrapper", "pynccl_wrapper", "patch_pynccl_wrapper"),
    ("fused_moe", "fused_moe", "patch_fused_moe_functions"),
    ("prefill_attention", "prefill_attention", "patch_prefill_attention"),
    ("fp32_tp_allreduce", "fp32_tp_allreduce", "patch_fp32_tp_allreduce"),
    ("triton_attention", "triton_attention", "patch_triton_attention"),
    ("qwen_vl_image_resize", "qwen_vl_image_resize", "patch_qwen_vl_image_resize"),
    ("mamba_extra_buffer", "mamba_extra_buffer", "patch_mamba_extra_buffer"),
    ("speculative_triton", "speculative_triton", "patch_speculative_triton"),
)

# Patches that only make sense when the vendor package they modify is installed.
# In Empty mode (no flashinfer / sgl_kernel) they are skipped instead of failing,
# and must not create shim modules that make the packages look importable.
# The sampler patch only changes the flashinfer sampling branch.
_PATCH_REQUIRES = {
    "flashinfer_utils_comm_backend": "flashinfer",
    "flashinfer_backend_classes": "flashinfer",
    "sampler": "flashinfer",
    "sgl_kernel_cutlass_scaled_mm_alias": "sgl_kernel",
}


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _load_patch(submodule: str, func: str):
    module = importlib.import_module(f"{__package__}.patches.{submodule}")
    return getattr(module, func)


def _run_required_patch(name: str, submodule: str, func: str) -> None:
    # RuntimeError (not ImportError) so the plugin loader does not mistake a broken
    # patch for an absent vendor module.
    try:
        _load_patch(submodule, func)()
    except Exception as error:
        raise RuntimeError(f"MetaX patch failed: {name}") from error


def _required_patches() -> tuple[tuple[str, str, str], ...]:
    return _PATCHES


def _apply_required_patches() -> None:
    applied, skipped = [], []
    for name, submodule, func in _required_patches():
        dep = _PATCH_REQUIRES.get(name)
        if dep is not None and not _has_module(dep):
            skipped.append(f"{name}(no {dep})")
            continue
        _run_required_patch(name, submodule, func)
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
