# CUDA vendor MRotaryEmbedding — delegates to SGLang's native implementation.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_cuda(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MRotaryEmbedding using SGLang's native CUDA/triton kernels.

    For 2D positions: calls forward_triton (triton_mrope_fused).
    For 1D positions: calls parent RotaryEmbedding.forward_cuda logic.
    """
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    # 1D positions: use standard sgl_kernel rope
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(obj, positions, query, key)


def mrotary_embedding_with_kv_cache_cuda(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    fused_set_kv_buffer_arg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused MRotaryEmbedding + KV cache write using SGLang's native CUDA kernels.

    For 2D positions with mrope_section: fused KV cache not supported,
    raises NotImplementedError.
    For 1D positions: delegates to RotaryEmbedding.forward_cuda with fused_args.
    """
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        raise NotImplementedError(
            "Fused RoPE + KV cache write is not supported for 2D multimodal positions. "
            "SGLang's triton_mrope_fused does not support fused_set_kv_buffer_arg."
        )
    # 1D positions: delegate to base RotaryEmbedding which supports fused KV write
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(
        obj,
        positions,
        query,
        key,
        fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
    )
