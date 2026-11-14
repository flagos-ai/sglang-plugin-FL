# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagOS implementations for FLA ops."""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def chunk_gated_delta_rule_flagos(
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
    """Run SGLang's chunk GDN contract with FlagGems FLA kernels."""
    from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd

    if head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)

    if initial_state_indices is not None:
        if initial_state is None:
            raise ValueError("initial_state_indices requires initial_state")
        valid_state = initial_state_indices >= 0
        selected_state = initial_state.new_zeros(
            (initial_state_indices.numel(), *initial_state.shape[1:])
        )
        if valid_state.any():
            selected_state[valid_state] = initial_state[
                initial_state_indices[valid_state]
            ]
        initial_state = selected_state

    if use_qk_l2norm_in_kernel:
        q = (q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
            q.dtype
        )
        k = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(
            k.dtype
        )

    # The low-level FlagGems API exposes the intermediate h tensor required by
    # SGLang's state-tracking path in addition to the output tensor.
    _, output, _, _, _, h, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    if head_first:
        output = output.transpose(1, 2)

    return output.to(q.dtype), None, h


def fused_recurrent_gated_delta_rule_flagos(
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
    """FlagOS implementation of fused_recurrent_gated_delta_rule."""
    from flag_gems.fused.FLA import fused_recurrent_gated_delta_rule_fwd

    return fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_flagos(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SGLang's packed GDN decode through the FlagGems recurrent kernel."""
    from flag_gems.fused.FLA import fused_recurrent_gated_delta_rule_fwd

    if mixed_qkv.ndim != 2:
        raise ValueError(f"mixed_qkv must be 2D, got {mixed_qkv.ndim}D")
    if initial_state.ndim != 4:
        raise ValueError(f"initial_state must be 4D, got {initial_state.ndim}D")

    batch = mixed_qkv.shape[0]
    num_value_heads, value_dim, key_dim = initial_state.shape[-3:]
    qk_dim = mixed_qkv.shape[-1] - num_value_heads * value_dim
    if qk_dim <= 0 or qk_dim % 2:
        raise ValueError("mixed_qkv has an invalid packed dimension")

    query_dim = qk_dim // 2
    if query_dim % key_dim:
        raise ValueError("packed query dimension is not divisible by key_dim")
    num_query_heads = query_dim // key_dim
    if num_query_heads <= 0 or num_value_heads % num_query_heads:
        raise ValueError("invalid query/value head configuration")

    query, key, value = torch.split(
        mixed_qkv,
        (query_dim, query_dim, num_value_heads * value_dim),
        dim=-1,
    )
    # query = query.view(batch, 1, num_query_heads, key_dim)
    # key = key.view(batch, 1, num_query_heads, key_dim)
    # value = value.view(batch, 1, num_value_heads, value_dim)

    query = query.view(1, batch, num_query_heads, key_dim)
    key = key.view(1, batch, num_query_heads, key_dim)
    value = value.view(1, batch, num_value_heads, value_dim)

    # Match SGLang's packed-decode gating math:
    # g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
    g = -torch.exp(A_log.float()) * F.softplus(
        a.float() + dt_bias.float(), beta=1.0, threshold=20.0
    )
    beta = torch.sigmoid(b.float()).to(b.dtype)
    # g = g.to(a.dtype).unsqueeze(1)
    # beta = beta.unsqueeze(1)

    g = g.to(a.dtype).unsqueeze(0)
    beta = beta.unsqueeze(0)
    # Packed decode contains one token for each independent request.  Describe
    # that layout explicitly because FlagGems uses cu_seqlens both to map the
    # flattened tokens to requests and to select their SSM cache slots.
    cu_seqlens = torch.arange(
        batch + 1,
        device=mixed_qkv.device,
        dtype=torch.long,
    )

    output, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # FlagGems keeps an NK dimension; packed decode's public contract does not.
    if output.ndim == out.ndim + 1 and output.shape[0] == 1:
        output = output.squeeze(0)
    # out.copy_(output.reshape_as(out))

    out.copy_(output.transpose(0, 1))
    return out, final_state