# MetaX FLA operator implementations.

from __future__ import annotations

from typing import Optional, Tuple

import torch

_chunk_gated_delta_rule_num_stages_patched = False


def _patch_chunk_gated_delta_rule_num_stages() -> None:
    """Force the chunk gated delta rule h kernel to launch with num_stages=1."""
    global _chunk_gated_delta_rule_num_stages_patched
    if _chunk_gated_delta_rule_num_stages_patched:
        return

    import triton

    from sglang.srt.layers.attention.fla import chunk as chunk_module
    from sglang.srt.layers.attention.fla import chunk_delta_h
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from sglang.srt.layers.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    def chunk_gated_delta_rule_fwd_h_metax(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        save_new_value: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, Hg, K, V = *k.shape, u.shape[-1]
        H = u.shape[-2]
        BT = CHUNK_SIZE

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is None:
            N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        else:
            N, NT, chunk_offsets = (
                len(cu_seqlens) - 1,
                len(chunk_indices),
                prepare_chunk_offsets(cu_seqlens, BT),
            )
        assert K <= 256, "current kernel does not support head dimension larger than 256."

        h = k.new_empty(B, NT, H, V, K)
        v_new = torch.empty_like(u) if save_new_value else None

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), N * H)

        chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BV=32,
            USE_G=g is not None,
            USE_GK=gk is not None,
            USE_INITIAL_STATE=initial_state is not None,
            INPLACE_UPDATE=True,
            SAVE_NEW_VALUE=v_new is not None,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
        )
        return h, v_new

    chunk_delta_h.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h_metax
    chunk_module.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h_metax
    _chunk_gated_delta_rule_num_stages_patched = True


def chunk_gated_delta_rule_metax(
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
    _patch_chunk_gated_delta_rule_num_stages()

    from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
        chunk_gated_delta_rule_cuda,
    )

    return chunk_gated_delta_rule_cuda(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_metax(
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
    from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
        fused_recurrent_gated_delta_rule_cuda,
    )

    return fused_recurrent_gated_delta_rule_cuda(
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


def fused_recurrent_gated_delta_rule_packed_decode_metax(
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
    from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
        fused_recurrent_gated_delta_rule_packed_decode_cuda,
    )

    return fused_recurrent_gated_delta_rule_packed_decode_cuda(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    )
