# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the native Ascend GDN prefill gating layout for SGLang 0.5.18."""

from functools import wraps
import importlib
from importlib import metadata
import logging

logger = logging.getLogger(__name__)

_GDN_TRITON_MODULE = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
_NATIVE_CHUNK_MODULE = "sgl_kernel_npu.fla.chunk"
_PATCH_MARKER = "_sglang_fl_ascend_batched_prefill_gating"


def patch_gdn_prefill_gating_layout() -> bool:
    """Match the sgl-kernel-npu 2026.9.0 prefill ABI."""

    try:
        if (
            metadata.version("sglang") != "0.5.18"
            or metadata.version("sgl-kernel-npu") != "2026.9.0"
        ):
            return False
    except metadata.PackageNotFoundError:
        return False

    try:
        gdn_triton = importlib.import_module(_GDN_TRITON_MODULE)
        native_chunk = importlib.import_module(_NATIVE_CHUNK_MODULE)
        original = gdn_triton.chunk_gated_delta_rule
    except (AttributeError, ImportError):
        logger.debug("SGLang GDN prefill kernel is unavailable", exc_info=True)
        return False

    if getattr(original, _PATCH_MARKER, False):
        return True
    if original is not getattr(native_chunk, "chunk_gated_delta_rule_npu", None):
        return False

    @wraps(original)
    def chunk_with_batched_gating(q, k, v, g, beta, *args, **kwargs):
        # SGLang's extend call passes options by keyword. The fused gating
        # producer returns [T, HV], but the native chunk expects [1, T, HV].
        if (
            not args
            and not kwargs.get("head_first", False)
            and v.ndim == 4
            and v.shape[0] == 1
            and g.shape == beta.shape == v.shape[1:3]
        ):
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)
        return original(q, k, v, g, beta, *args, **kwargs)

    setattr(chunk_with_batched_gating, _PATCH_MARKER, True)
    gdn_triton.chunk_gated_delta_rule = chunk_with_batched_gating
    logger.info("Adapted the Ascend native GDN prefill gating layout")
    return True
