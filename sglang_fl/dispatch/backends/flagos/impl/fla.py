# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagOS implementations for FLA ops."""

from typing import Optional, Tuple
import torch

# Global flag to ensure triton allocator is only set once
_TRITON_ALLOCATOR_INSTALLED = False


def _ensure_triton_allocator(device: torch.device) -> None:
    """Set up triton allocator if not already done.

    This is required for FlagGems FLA kernels that use triton runtime memory allocation.
    Pattern follows flag_gems.fused.moe_align_block_size._install_triton_default_allocator.
    """
    global _TRITON_ALLOCATOR_INSTALLED
    if _TRITON_ALLOCATOR_INSTALLED:
        return

    try:
        import triton

        def _alloc(size: int, alignment: int, stream: Optional[int]):
            return torch.empty((size,), dtype=torch.uint8, device=device).data_ptr()

        triton.set_allocator(_alloc)
        _TRITON_ALLOCATOR_INSTALLED = True
    except ImportError:
        # triton not available, kernels will fail anyway
        pass


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

    _ensure_triton_allocator(q.device)

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

    _ensure_triton_allocator(q.device)

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """flagos vendor implementation - uses SGLang's native triton kernel for packed decode."""
    from flaggems_sglang import fused_recurrent_gated_delta_rule_packed_decode as fused_recurrent_gated_delta_rule_packed_decode_flagos
    
    return fused_recurrent_gated_delta_rule_packed_decode_flagos(
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
