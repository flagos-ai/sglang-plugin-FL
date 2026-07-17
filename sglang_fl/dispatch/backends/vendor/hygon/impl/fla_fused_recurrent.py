# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge for FLA fused_recurrent_gated_delta_rule function."""

from typing import Optional, Tuple
import torch

def fused_recurrent_gated_delta_rule_hcu(
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

    from sglang.srt.layers.attention.fla.fused_recurrent import (
        FusedRecurrentFunction,
    )

    return FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
