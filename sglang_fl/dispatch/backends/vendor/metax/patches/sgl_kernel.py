"""Patch W8A8 int8 cutlass_scaled_mm registration for MetaX."""

import importlib
import importlib.util
import logging
import sys
import types
from typing import Optional

logger = logging.getLogger(__name__)

_PATCHED_MARKER = "__sglang_fl_metax_cutlass_scaled_mm_alias__"
_CUTLASS_SCALED_MM = "cutlass_scaled_mm"
_INT8_SCALED_MM = "int8_scaled_mm"
_W8A8_INT8_MODULE = "sglang.srt.layers.quantization.w8a8_int8"

_LOADED_MODULE_GLOBALS = {
    "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8": (
        _INT8_SCALED_MM,
    ),
}


def _get_or_create_sgl_kernel_module() -> types.ModuleType:
    module = sys.modules.get("sgl_kernel")
    if isinstance(module, types.ModuleType):
        return module

    if importlib.util.find_spec("sgl_kernel") is not None:
        try:
            module = importlib.import_module("sgl_kernel")
            if isinstance(module, types.ModuleType):
                return module
        except Exception:
            logger.debug("metax sgl_kernel patch: real import failed", exc_info=True)

    module = types.ModuleType("sgl_kernel")
    module.__loader__ = None
    module.__file__ = "<sglang_fl_metax_patch>"
    module.__fl_sgl_kernel_shim_only__ = True
    sys.modules["sgl_kernel"] = module
    return module


def _register_cutlass_scaled_mm_op(raw_cutlass_scaled_mm):
    import torch

    from sglang.srt.utils.common import direct_register_custom_op

    def cutlass_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return raw_cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    def cutlass_scaled_mm_fake(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a.shape[-2]
        n = b.shape[-1]
        return a.new_empty((m, n), dtype=out_dtype)

    direct_register_custom_op(
        op_name=_CUTLASS_SCALED_MM,
        op_func=cutlass_scaled_mm,
        mutates_args=[],
        fake_impl=cutlass_scaled_mm_fake,
    )
    return torch.ops.sglang.cutlass_scaled_mm


def patch_sgl_kernel_cutlass_scaled_mm_alias() -> None:
    """Register and route the W8A8 int8 ``cutlass_scaled_mm`` op.

    This mirrors the local SGLang source edit that imports
    ``sgl_kernel.cutlass_scaled_mm`` as ``sgl_cutlass_scaled_mm``, registers
    ``torch.ops.sglang.cutlass_scaled_mm``, and lets the original
    ``from sgl_kernel import int8_scaled_mm`` import receive that registered op
    without editing SGLang source files.
    """
    sgl_kernel = _get_or_create_sgl_kernel_module()
    raw_cutlass_scaled_mm = getattr(sgl_kernel, _CUTLASS_SCALED_MM, None)
    if raw_cutlass_scaled_mm is None:
        logger.debug(
            "metax sgl_kernel patch skipped: sgl_kernel.%s is not available",
            _CUTLASS_SCALED_MM,
        )
        return

    replacement = _register_cutlass_scaled_mm_op(raw_cutlass_scaled_mm)
    setattr(sgl_kernel, _CUTLASS_SCALED_MM, replacement)
    setattr(sgl_kernel, _INT8_SCALED_MM, replacement)
    setattr(sgl_kernel, _PATCHED_MARKER, True)

    w8a8_int8 = sys.modules.get(_W8A8_INT8_MODULE)
    if isinstance(w8a8_int8, types.ModuleType):
        setattr(w8a8_int8, _CUTLASS_SCALED_MM, replacement)
        setattr(w8a8_int8, _INT8_SCALED_MM, replacement)

    for module_name, symbols in _LOADED_MODULE_GLOBALS.items():
        loaded_module = sys.modules.get(module_name)
        if not isinstance(loaded_module, types.ModuleType):
            continue
        for symbol in symbols:
            if hasattr(loaded_module, symbol):
                setattr(loaded_module, symbol, replacement)

    logger.info(
        "patched %s through torch.ops.sglang.%s",
        _CUTLASS_SCALED_MM,
        _CUTLASS_SCALED_MM,
    )
