# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Enable the MTT S5000 custom all-reduce + residual/RMSNorm path.

The MUSA vendor runtime carries a JIT custom all-reduce implementation that is
newer than the SGLang revision used by the FlagOS image.  Keep the integration
in the plugin: replace the custom-allreduce class before TP groups are created,
then reuse SGLang's existing ``forward_with_allreduce_fusion`` seam.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_CUSTOM_AR_FUSED_RMSNORM"
_patch_applied = False


def _device_is_s5000() -> bool:
    try:
        import torch

        return "S5000" in str(torch.musa.get_device_name()).upper()
    except Exception:
        return False


def _enabled() -> bool:
    value = os.environ.get(_ENV_NAME, "auto").strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    if value != "auto":
        raise ValueError(f"Unsupported {_ENV_NAME} value: {value!r}")
    return _device_is_s5000()


def _dispatch_musa_custom_allreduce():
    from sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.communicator import (
        MusaJitCustomAllreduce,
    )

    return MusaJitCustomAllreduce


def _select_group(use_attn_tp_group: bool):
    from sglang.srt.distributed.parallel_state import (
        get_attn_tp_group,
        get_moe_ep_group,
        get_moe_tp_group,
        get_tp_group,
    )

    if use_attn_tp_group:
        return get_attn_tp_group()
    moe_ep_group = get_moe_ep_group()
    if moe_ep_group.world_size > 1:
        return moe_ep_group
    moe_tp_group = get_moe_tp_group()
    if moe_tp_group.world_size > 1:
        return moe_tp_group
    return get_tp_group()


def _forward_with_musa_allreduce_fusion(
    norm_module,
    x,
    residual,
    post_residual_addition,
    weight,
    use_attn_tp_group: bool = True,
):
    """Try the communicator-native fused path, with exact generic fallback."""
    if residual is None:
        return norm_module.forward(x, residual, post_residual_addition)

    group = _select_group(use_attn_tp_group)
    if group.world_size <= 1:
        return norm_module.forward(x, residual, post_residual_addition)

    if post_residual_addition is not None:
        residual = residual + post_residual_addition

    fused_result = group.fused_allreduce_rmsnorm(
        x, residual, weight, norm_module.variance_epsilon
    )
    if fused_result is not None:
        return fused_result

    # Unsupported shapes, missing JIT dependencies, or an explicitly disabled
    # communicator retain the pre-patch all-reduce then RMSNorm semantics.
    x = group.all_reduce(x)
    return norm_module.forward(x, residual, None)


def _musa_fusion_gate(original_gate, batch_size: int) -> bool:
    # The fused implementation performs the full dtype/shape/size checks.  This
    # gate only avoids routing empty or impossible row counts through it.
    if 0 < int(batch_size) <= 131072:
        return True
    return bool(original_gate(batch_size))


def apply_musa_custom_allreduce_rmsnorm_patch() -> None:
    global _patch_applied
    if _patch_applied:
        return
    if not _enabled():
        logger.info("MUSA custom AR + RMSNorm patch disabled by %s", _ENV_NAME)
        return

    try:
        from sglang.srt.distributed import parallel_state
        from sglang.srt.distributed.device_communicators import custom_all_reduce
        from sglang.srt.layers import communicator, layernorm
    except Exception as exc:
        logger.warning("MUSA custom AR + RMSNorm patch skipped: %s", exc)
        return

    # GroupCoordinator imports this symbol into parallel_state, so update both
    # bindings before any TP/attention/MoE group is constructed.
    custom_all_reduce.dispatch_custom_allreduce = _dispatch_musa_custom_allreduce
    parallel_state.dispatch_custom_allreduce = _dispatch_musa_custom_allreduce
    layernorm._forward_with_allreduce_fusion = _forward_with_musa_allreduce_fusion

    original_gate = communicator.apply_flashinfer_allreduce_fusion

    def gate_with_musa(batch_size: int) -> bool:
        return _musa_fusion_gate(original_gate, batch_size)

    communicator.apply_flashinfer_allreduce_fusion = gate_with_musa
    _patch_applied = True
    logger.info(
        "MUSA S5000 JIT custom all-reduce + residual/RMSNorm patch applied "
        "(graph registered-input mode defaults off for SGLang compatibility)"
    )


__all__ = ["apply_musa_custom_allreduce_rmsnorm_patch"]
