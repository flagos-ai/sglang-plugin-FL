# Copyright (c) 2026 BAAI. All rights reserved.
"""CUDA vendor implementations for FLA ops (fallback to SGLang native triton)."""

from typing import Optional, Tuple
import torch


def _get_native_fla_op(name: str):
    """Return the unpatched SGLang FLA function saved by ``fla_patch``."""
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(name)
    if fn is None:
        raise RuntimeError(
            f"SGLang native FLA op '{name}' was not captured before dispatch patching"
        )
    return fn


def chunk_gated_delta_rule_cuda(
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
    """CUDA vendor implementation - uses SGLang's native triton kernel."""
    native_op = _get_native_fla_op("chunk_gated_delta_rule")
    return native_op(
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


def fused_recurrent_gated_delta_rule_cuda(
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
    """CUDA vendor implementation - uses SGLang's native triton kernel."""
    # SGLang 0.5.18 moved FLA into ``sglang.kernels`` and its public
    # recurrent function does not accept the two legacy scheduling hints.
    # They are retained in the plugin bridge for older backends, but the
    # captured native function is the source of truth for CUDA.
    del ssm_state_indices, num_accepted_tokens
    native_op = _get_native_fla_op("fused_recurrent_gated_delta_rule")
    return native_op(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_cuda(
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
    """CUDA vendor implementation - uses SGLang's native triton kernel for packed decode."""
    native_op = _get_native_fla_op(
        "fused_recurrent_gated_delta_rule_packed_decode"
    )
    return native_op(
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
