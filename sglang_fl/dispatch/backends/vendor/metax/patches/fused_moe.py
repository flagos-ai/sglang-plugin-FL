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


def moe_align_block_size_flaggems(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlagGems replacement for sgl_kernel.moe_align_block_size (Empty mode).

    Same outputs as SGLang's wrapper: sorted_ids padded with ``topk_ids.numel()``,
    per-block expert ids and the padded token count. SGLang passes
    ``num_experts + 1`` to sgl_kernel to cover EP-filtered ``-1`` expert ids;
    FlagGems does not handle ``-1``, so this path is only valid without EP.
    """
    if ignore_invalid_expert:
        raise NotImplementedError(
            "FlagGems moe_align_block_size does not support ignore_invalid_expert"
        )
    from flag_gems import moe_align_block_size

    return moe_align_block_size(topk_ids, block_size, num_experts)


def _patch_moe_align_block_size() -> bool:
    """Use FlagGems moe_align_block_size when sgl_kernel is not installed.

    SGLANG_FL_METAX_MOE_ALIGN=flaggems forces it even when sgl_kernel exists
    (used to align NonEmpty with Empty in performance comparisons).
    """
    import importlib.util
    import os

    forced = (
        os.environ.get("SGLANG_FL_METAX_MOE_ALIGN", "").strip().lower() == "flaggems"
    )
    if not forced:
        try:
            if importlib.util.find_spec("sgl_kernel") is not None:
                return False
        except (ImportError, ValueError):
            pass

    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe as fused_moe_module,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        moe_align_block_size as align_module,
    )

    align_module.moe_align_block_size = moe_align_block_size_flaggems
    if hasattr(fused_moe_module, "moe_align_block_size"):
        fused_moe_module.moe_align_block_size = moe_align_block_size_flaggems
    return True


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

    align_patched = _patch_moe_align_block_size()

    _fused_moe_functions_patched = True
    logger.info(
        "patched MetaX fused_moe helpers: inplace_fused_experts, "
        "outplace_fused_experts, fused_experts, silu_and_mul, gelu_and_mul, "
        "moe_sum_reduce%s",
        ", moe_align_block_size(flag_gems)" if align_patched else "",
    )
