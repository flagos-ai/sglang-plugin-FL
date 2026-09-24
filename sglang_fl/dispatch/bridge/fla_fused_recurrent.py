# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge for FLA fused_recurrent_gated_delta_rule function."""

from typing import Optional, Tuple
import torch

from sglang_fl.dispatch import call_op


def fused_recurrent_gated_delta_rule_bridge(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Bridge function for fused_recurrent_gated_delta_rule.

    Dispatches to backend implementations via call_op().

    Args:
        q: queries of shape [B, T, H, K]
        k: keys of shape [B, T, H, K]
        v: values of shape [B, T, HV, V]
        g: gating (decays) of shape [B, T, HV]
        beta: betas of shape [B, T, HV]
        scale: scale factor (default: 1/sqrt(K))
        initial_state: initial state of shape [N, HV, V, K]
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths
        ssm_state_indices: indices to map sequences to states
        num_accepted_tokens: number of accepted tokens per sequence

    Returns:
        o: output of shape [B, T, HV, V]
        final_state: final state of shape [N, HV, V, K]
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    return call_op(
        "fused_recurrent_gated_delta_rule",
        q=q,
        k=k,
        v=v,
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
