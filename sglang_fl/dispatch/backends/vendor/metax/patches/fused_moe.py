"""Patch SGLang FusedMoE helpers for MetaX."""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_fused_moe_functions_patched = False


def _apply_expert_filter(
    result: torch.Tensor,
    expert_ids: torch.Tensor | None,
    expert_step: int,
) -> torch.Tensor:
    if expert_ids is None:
        return result

    row_ids = torch.arange(result.shape[0], device=result.device)
    active = expert_ids[row_ids // expert_step] != -1
    return result * active.to(result.dtype).unsqueeze(-1)


def silu_and_mul_python(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_step: int = 1,
) -> torch.Tensor:
    hidden_size = input.shape[-1] // 2
    result = F.silu(input[..., :hidden_size]) * input[..., hidden_size:]
    result = _apply_expert_filter(result, expert_ids, expert_step)
    if out is not None:
        out.copy_(result)
        return out
    return result


def gelu_and_mul_python(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_step: int = 1,
) -> torch.Tensor:
    hidden_size = input.shape[-1] // 2
    result = F.gelu(input[..., :hidden_size]) * input[..., hidden_size:]
    result = _apply_expert_filter(result, expert_ids, expert_step)
    if out is not None:
        out.copy_(result)
        return out
    return result


def moe_sum_reduce_triton_metax(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        moe_sum_reduce_triton,
    )

    return moe_sum_reduce_triton(input, output, routed_scaling_factor)


def patch_fused_moe_functions() -> None:
    """Replace SGLang FusedMoE helpers with MetaX-compatible functions."""
    global _fused_moe_functions_patched
    if _fused_moe_functions_patched:
        return

    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe as fused_moe_module,
    )

    original_inplace_fused_experts = fused_moe_module.inplace_fused_experts
    original_outplace_fused_experts = fused_moe_module.outplace_fused_experts
    original_fused_experts = fused_moe_module.fused_experts

    @functools.wraps(original_inplace_fused_experts)
    def inplace_fused_experts_metax(*args, **kwargs):
        # TODO(metax): adjust inplace fused experts arguments here if needed.
        return original_inplace_fused_experts(*args, **kwargs)

    @functools.wraps(original_outplace_fused_experts)
    def outplace_fused_experts_metax(*args, **kwargs):
        # TODO(metax): adjust outplace fused experts arguments here if needed.
        return original_outplace_fused_experts(*args, **kwargs)

    @functools.wraps(original_fused_experts)
    def fused_experts_metax(*args, **kwargs):
        # TODO(metax): adjust high-level fused_experts behavior here if needed.
        return original_fused_experts(*args, **kwargs)

    fused_moe_module.inplace_fused_experts = inplace_fused_experts_metax
    fused_moe_module.outplace_fused_experts = outplace_fused_experts_metax
    fused_moe_module.fused_experts = fused_experts_metax
    fused_moe_module.silu_and_mul = silu_and_mul_python
    fused_moe_module.gelu_and_mul = gelu_and_mul_python
    fused_moe_module.moe_sum_reduce = moe_sum_reduce_triton_metax

    _fused_moe_functions_patched = True
    logger.info(
        "patched MetaX fused_moe helpers: inplace_fused_experts, "
        "outplace_fused_experts, fused_experts, silu_and_mul, gelu_and_mul, "
        "moe_sum_reduce"
    )
