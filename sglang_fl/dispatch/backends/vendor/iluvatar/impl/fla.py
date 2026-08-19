# Copyright (c) 2026 BAAI. All rights reserved.
"""Iluvatar vendor implementations for FLA ops."""

from typing import Optional, Tuple

import torch


def _require_fla_original(name: str):
    """Return patched-away SGLang FLA fn, or raise NotImplementedError for fallback."""
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(name)
    if fn is None:
        raise NotImplementedError(
            f"FLA original '{name}' not available; falling back to flaggems/reference"
        )
    return fn


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
    raise NotImplementedError(
        "chunk_gated_delta_rule_iluvatar; falling back to flaggems/reference"
    )


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
    """Iluvatar vendor implementation."""
    native_fn = _require_fla_original("fused_recurrent_gated_delta_rule")
    return native_fn(
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
    """Iluvatar vendor implementation."""
    native_fn = _require_fla_original(
        "fused_recurrent_gated_delta_rule_packed_decode"
    )
    return native_fn(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
