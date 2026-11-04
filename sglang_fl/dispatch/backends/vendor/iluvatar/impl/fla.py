# Copyright (c) 2026 BAAI. All rights reserved.
"""Iluvatar's independent GDN kernels with the SGLang 0.5.18 public contracts."""

from typing import Optional, Tuple

import torch


def _varlen_metadata(q, v, initial_state, state_indices, cu_seqlens):
    """Materialize the state pool and varlen offsets the recurrence indexes."""
    if cu_seqlens is None:
        length = q.shape[1]
        cu_seqlens = (
            torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device) * length
        )
    elif q.shape[0] != 1:
        raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
    sequences = cu_seqlens.numel() - 1
    state_shape = (sequences, v.shape[-2], v.shape[-1], q.shape[-1])
    if initial_state is None:
        if state_indices is not None:
            raise ValueError("state indices require initial_state")
        initial_state = q.new_zeros(state_shape, dtype=torch.float32)
    elif initial_state.ndim != 4 or initial_state.shape[-3:] != state_shape[-3:]:
        raise ValueError(f"initial_state must have trailing shape {state_shape[-3:]}")
    if state_indices is None:
        if initial_state.shape[0] != sequences:
            raise ValueError(f"initial_state must have shape {state_shape}")
        state_indices = torch.arange(sequences, dtype=torch.int32, device=q.device)
    elif state_indices.shape != (sequences,):
        raise ValueError("state indices must have one entry per sequence")
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


def _chunk_checkpoints(checkpoints, shape, key_dim, cu_seqlens):
    """Expose precisely the chunk history allocated by upstream 0.5.18."""
    batch, length, heads, value_dim = shape
    if checkpoints.ndim != 5 or checkpoints.shape[-3:] != (heads, value_dim, key_dim):
        raise RuntimeError("Iluvatar GDN returned an invalid checkpoint layout")
    if cu_seqlens is None:
        chunks_per_sequence = (length + 63) // 64
        count = batch * chunks_per_sequence
        output_shape = (batch, chunks_per_sequence, heads, value_dim, key_dim)
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        count = int(((lengths + 63) // 64).sum().item())
        output_shape = (1, count, heads, value_dim, key_dim)
    # The independent recurrence reserves a safe upper bound for varlen
    # checkpoint storage. Its unused capacity is not part of the public ABI.
    if checkpoints.shape[0] != 1 or checkpoints.shape[1] < count:
        raise RuntimeError("Iluvatar GDN returned an invalid checkpoint buffer")
    return checkpoints[:, :count].reshape(output_shape)


def chunk_gated_delta_rule_iluvatar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Preserve indexed pool updates; unindexed calls own a private state copy.

    The unindexed form is our convenience extension. Official SGLang 0.5.18
    invokes the chunk kernel with an explicit slot map and updates that pool.
    """
    if head_first:
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        g, beta = g.transpose(1, 2), beta.transpose(1, 2)

    output_shape = v.shape
    input_cu_seqlens = cu_seqlens
    # Qwen's indexed cache path keeps the original pool without an extra copy.
    # The unindexed extension must not expose the kernel's in-place workspace.
    working_state = (
        initial_state.clone(memory_format=torch.contiguous_format)
        if initial_state is not None and initial_state_indices is None
        else initial_state
    )
    state, indices, cu_seqlens = _varlen_metadata(
        q, v, working_state, initial_state_indices, cu_seqlens
    )
    scale = k.shape[-1] ** -0.5 if scale is None else float(scale)
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
    checkpoints = _chunk_checkpoints(
        checkpoints, output_shape, state.shape[-1], input_cu_seqlens
    )
    if head_first:
        output = output.transpose(1, 2)
    # Upstream chunk returns no final-state tensor even for an unpooled call.
    # Pooled cache updates are owned by the recurrence; h contains chunk starts.
    return output, None, checkpoints


def fused_recurrent_gated_delta_rule_iluvatar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if num_accepted_tokens is not None:
        raise NotImplementedError("speculative decoding state rollback is unsupported")

    # The kernel updates its state buffer in place. The public recurrent API
    # reads initial_state without changing it and always returns FP32 final state.
    working_state = (
        initial_state.to(dtype=torch.float32, copy=True).contiguous()
        if initial_state is not None
        else None
    )
    state, indices, cu_seqlens = _varlen_metadata(
        q, v, working_state, ssm_state_indices, cu_seqlens
    )
    output_shape = v.shape
    scale = k.shape[-1] ** -0.5 if scale is None else float(scale)
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
