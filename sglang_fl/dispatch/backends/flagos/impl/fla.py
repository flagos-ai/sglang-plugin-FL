# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagOS implementations for FLA ops."""

from typing import Optional, Tuple
import torch


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
    """FlagOS implementation of chunk_gated_delta_rule.

    Mirrors sglang's ChunkGatedDeltaRuleFunction.forward but calls
    FlagGems' chunk_gated_delta_rule_fwd as the compute backend.
    """
    from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd
    from flag_gems.fused.chunk_gated_delta_rule import _l2_normalize_last_dim

    if initial_state is not None and initial_state_indices is not None:
        initial_state = initial_state[initial_state_indices]

    if use_qk_l2norm_in_kernel:
        q = _l2_normalize_last_dim(q)
        k = _l2_normalize_last_dim(k)

    # FlagGems fwd returns 7 values: g, o, A, final_state, w, h, v_new
    # sglang expects (o, h) from forward
    _, o, _, _, _, h, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return o.to(q.dtype), None, h


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
