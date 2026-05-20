# MUSA FLA (Flash Linear Attention) operator implementations.
#
# TODO: Replace NotImplementedError with torch_musa native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

from typing import Optional, Tuple

import torch


def chunk_gated_delta_rule_musa(
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
    """chunk_gated_delta_rule — not yet implemented on MUSA."""
    raise NotImplementedError(
        "chunk_gated_delta_rule not yet implemented on MUSA"
    )


def fused_recurrent_gated_delta_rule_musa(
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
    """fused_recurrent_gated_delta_rule — not yet implemented on MUSA."""
    raise NotImplementedError(
        "fused_recurrent_gated_delta_rule not yet implemented on MUSA"
    )


def fused_recurrent_gated_delta_rule_packed_decode_musa(
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
    """fused_recurrent_gated_delta_rule_packed_decode — not yet implemented on MUSA."""
    raise NotImplementedError(
        "fused_recurrent_gated_delta_rule_packed_decode not yet implemented on MUSA"
    )
