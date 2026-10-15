# Copyright (c) 2026 BAAI. All rights reserved.
"""Hygon chunk FLA adapter for SGLang's public kernel contract."""

from typing import Optional

import torch

from .fla_compat import original_fla


def chunk_gated_delta_rule_hcu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Preserve public validation, indexed state updates and the return tuple.

    SGLang 0.5.18 moved FLA into ``sglang.kernels``. Calling the saved public
    function also avoids depending on the private autograd Function ABI.
    """
    return original_fla("chunk_gated_delta_rule")(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
