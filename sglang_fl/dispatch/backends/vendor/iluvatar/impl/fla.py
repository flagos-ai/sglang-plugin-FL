# Copyright (c) 2026 BAAI. All rights reserved.

from typing import Optional, Tuple

import torch


def _varlen_metadata(q, v, initial_state, state_indices, cu_seqlens):
    """Materialize the state pool and varlen offsets the recurrence indexes."""
    if cu_seqlens is None:
        length = q.shape[1]
        cu_seqlens = torch.arange(
            0,
            (q.shape[0] + 1) * length,
            length,
            dtype=torch.int32,
            device=q.device,
        )
    sequences = cu_seqlens.numel() - 1
    if initial_state is None:
        initial_state = q.new_zeros(sequences, v.shape[-2], v.shape[-1], q.shape[-1])
    if state_indices is None:
        state_indices = torch.arange(sequences, dtype=torch.int32, device=q.device)
    return (
        initial_state,
        state_indices.to(torch.int32).contiguous(),
        cu_seqlens.to(torch.int32).contiguous(),
    )


def _flatten_tokens(q, k, v, g, beta):
    return (
        q.reshape(-1, q.shape[-2], q.shape[-1]),
        k.reshape(-1, k.shape[-2], k.shape[-1]),
        v.reshape(-1, v.shape[-2], v.shape[-1]),
        g.reshape(-1, g.shape[-1]),
        beta.reshape(-1, beta.shape[-1]),
    )


def chunk_gated_delta_rule_iluvatar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    if head_first:
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        g, beta = g.transpose(1, 2), beta.transpose(1, 2)

    # A pooled state carries its own slot map; only an unpooled call owns the
    # final state and gets it back.
    return_state = initial_state_indices is None
    state, indices, cu_seqlens = _varlen_metadata(
        q, v, initial_state, initial_state_indices, cu_seqlens
    )
    output_shape = v.shape
    q, k, v, g, beta = _flatten_tokens(q, k, v, g, beta)

    from .gdn_triton import recurrent_gdn

    output, checkpoints = recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        state,
        indices,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
        return_checkpoints=True,
    )
    output = output.view(output_shape)
    if head_first:
        output = output.transpose(1, 2)
    return output, state if return_state else None, checkpoints


def fused_recurrent_gated_delta_rule_iluvatar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if num_accepted_tokens is not None:
        raise NotImplementedError("speculative decoding state rollback is unsupported")

    state, indices, cu_seqlens = _varlen_metadata(
        q, v, initial_state, ssm_state_indices, cu_seqlens
    )
    output_shape = v.shape
    q, k, v, g, beta = _flatten_tokens(q, k, v, g, beta)

    from .gdn_triton import recurrent_gdn

    output = recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        state,
        indices,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
    )
    return output.view(output_shape), state if output_final_state else None


def fused_recurrent_gated_delta_rule_packed_decode_iluvatar(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .gdn_triton import packed_decode_gdn

    return packed_decode_gdn(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices.to(torch.int32).contiguous(),
        use_qk_l2norm_in_kernel,
    )
