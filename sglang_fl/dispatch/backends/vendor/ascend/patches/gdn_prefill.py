# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the native Ascend GDN kernel layouts for SGLang 0.5.18."""

from functools import wraps
import importlib
from importlib import metadata
import logging

logger = logging.getLogger(__name__)

_GDN_TRITON_MODULE = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
_NATIVE_CHUNK_MODULE = "sgl_kernel_npu.fla.chunk"
_PATCH_MARKER = "_sglang_fl_ascend_batched_prefill_gating"


def uses_native_vk_gdn_state() -> bool:
    """Identify the native recurrent ABI that stores each state in V,K order."""

    try:
        return (
            metadata.version("sglang") == "0.5.18"
            and metadata.version("sgl-kernel-npu") == "2026.9.0"
        )
    except metadata.PackageNotFoundError:
        return False


def patch_gdn_layout() -> bool:
    """Match the sgl-kernel-npu 2026.9.0 gating and recurrent-state layouts."""

    if not uses_native_vk_gdn_state():
        return False

    try:
        gdn_triton = importlib.import_module(_GDN_TRITON_MODULE)
        native_chunk = importlib.import_module(_NATIVE_CHUNK_MODULE)
        original = gdn_triton.chunk_gated_delta_rule
        kernel = gdn_triton.TritonGDNKernel
        original_extend = kernel.extend
        original_decode = kernel.decode
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

    @wraps(original_extend)
    def extend_with_native_state_layout(
        self, q, k, v, g, beta, *, ssm_states, **kwargs
    ):
        # Ordinary decode consumes the physical V,K pool. A speculative pool
        # already exposes a K,V view of that same V,K physical layout.
        physical_vk = ssm_states.is_contiguous() and ssm_states.shape[-2:] == (
            v.shape[-1],
            k.shape[-1],
        )
        chunk_state = ssm_states.transpose(-1, -2) if physical_vk else ssm_states
        # The upstream extend gathers selected rows before the native chunk
        # makes its K,V inputs contiguous.
        out, last_state, h = original_extend(
            self, q, k, v, g, beta, ssm_states=chunk_state, **kwargs
        )
        if physical_vk:
            # SGLang writes both tensors into the same pool (h for radix
            # tracking), so ordinary prefill results must also use V,K.
            if last_state is not None:
                last_state = last_state.transpose(-1, -2)
            if h is not None:
                h = h.transpose(-1, -2)
        return out, last_state, h

    @wraps(original_decode)
    def decode_with_native_state_layout(self, q, k, v, a, b, *, ssm_states, **kwargs):
        if not ssm_states.is_contiguous() and ssm_states.shape[-2:] == (
            k.shape[-1],
            v.shape[-1],
        ):
            physical_state = ssm_states.transpose(-1, -2)
            if physical_state.is_contiguous():
                # The native input guard makes arguments contiguous. Pass
                # the physical V,K view to keep decode's writes in the pool.
                ssm_states = physical_state
        return original_decode(self, q, k, v, a, b, ssm_states=ssm_states, **kwargs)

    setattr(chunk_with_batched_gating, _PATCH_MARKER, True)
    gdn_triton.chunk_gated_delta_rule = chunk_with_batched_gating
    kernel.extend = extend_with_native_state_layout
    kernel.decode = decode_with_native_state_layout
    logger.info("Adapted the Ascend native GDN gating and recurrent-state layouts")
    return True
