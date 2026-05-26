# Ascend FLA (Flash Linear Attention) operator implementations.
#
# chunk_gated_delta_rule: sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu
# fused_recurrent_* : never called on NPU — AscendGDNAttnBackend routes around them.

from __future__ import annotations

from typing import Optional, Tuple

import torch


def chunk_gated_delta_rule_ascend(
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
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu

    # chunk_gated_delta_rule_npu does not accept initial_state_indices; pre-index here as sglang does.
    if initial_state is not None and initial_state_indices is not None:
        initial_state = initial_state[initial_state_indices]

    return chunk_gated_delta_rule_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
