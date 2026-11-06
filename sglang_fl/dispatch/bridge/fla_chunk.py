# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge for FLA chunk_gated_delta_rule function."""

from typing import Optional
import torch

from sglang_fl.dispatch import call_op


def chunk_gated_delta_rule_bridge(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    inplace_update: bool = True,
):
    """
    Bridge function for chunk_gated_delta_rule.

    Dispatches to backend implementations via call_op().
    Signature matches sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    kwargs = dict(
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
    if q.device.type == "musa":
        kwargs["inplace_update"] = inplace_update
    elif not inplace_update:
        # Other vendor backends still expose the older signature. Preserve the
        # new SGLang semantics without passing an unknown kwarg to those backends.
        from sglang_fl.dispatch.fla_patch import get_original

        return get_original("chunk_gated_delta_rule")(**kwargs, inplace_update=False)
    return call_op("chunk_gated_delta_rule", **kwargs)
