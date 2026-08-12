from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T", "total_work_items"])
def _chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_SPC: tl.constexpr,
    NV: tl.constexpr,
    total_work_items,
):
    pid = tl.program_id(0)
    log2_e = 1.4426950408889634

    for work_id in range(pid, total_work_items, NUM_SPC):
        i_v = work_id % NV
        i_nh = work_id // NV
        i_n = i_nh // H
        i_h = i_nh % H

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            local_t = eos - bos
            nt = tl.cdiv(local_t, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            bos = i_n * T
            local_t = T
            nt = tl.cdiv(T, BT)
            boh = i_n * nt

        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

        h_base = h + ((boh * H + i_h) * V * K).to(tl.int64)
        v_base = v + ((bos * H + i_h) * V).to(tl.int64)
        k_base = k + ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
        w_base = w + ((bos * H + i_h) * K).to(tl.int64)
        if SAVE_NEW_VALUE:
            v_new_base = v_new + ((bos * H + i_h) * V).to(tl.int64)
        stride_v = H * V
        stride_h = H * V * K
        stride_k = Hg * K
        stride_w = H * K

        if USE_INITIAL_STATE:
            index = tl.load(initial_state_indices + i_n).to(tl.int32)
            h0 = initial_state + (index * stride_h + i_h * V * K).to(tl.int64)
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        for i_t in range(nt):
            p_h1 = tl.make_block_ptr(h_base + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h_base + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(h_base + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(h_base + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w_base, (local_t, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w_base, (local_t, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w_base, (local_t, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w_base, (local_t, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))

            p_v = tl.make_block_ptr(v_base, (local_t, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(v_new_base, (local_t, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

            last_idx = tl.minimum((i_t + 1) * BT, local_t) - 1
            if USE_G:
                b_g_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (local_t,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_v = b_v * tl.math.exp2((b_g_last - b_g) * log2_e)[:, None]
                b_g_last_exp = tl.math.exp2(b_g_last * log2_e)
                b_h1 = b_h1 * b_g_last_exp
                if K > 64:
                    b_h2 = b_h2 * b_g_last_exp
                if K > 128:
                    b_h3 = b_h3 * b_g_last_exp
                if K > 192:
                    b_h4 = b_h4 * b_g_last_exp

            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
                b_h1 *= tl.math.exp2(b_gk_last1 * log2_e)[None, :]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k2, mask=o_k2 < K, other=0.0).to(tl.float32)
                    b_h2 *= tl.math.exp2(b_gk_last2 * log2_e)[None, :]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k3, mask=o_k3 < K, other=0.0).to(tl.float32)
                    b_h3 *= tl.math.exp2(b_gk_last3 * log2_e)[None, :]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k4, mask=o_k4 < K, other=0.0).to(tl.float32)
                    b_h4 *= tl.math.exp2(b_gk_last4 * log2_e)[None, :]

            b_v = b_v.to(k.dtype.element_ty)
            p_k = tl.make_block_ptr(k_base, (K, local_t), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.trans(tl.dot(b_k, b_v))
            if K > 64:
                p_k = tl.make_block_ptr(k_base, (K, local_t), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            if K > 128:
                p_k = tl.make_block_ptr(k_base, (K, local_t), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            if K > 192:
                p_k = tl.make_block_ptr(k_base, (K, local_t), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.trans(tl.dot(b_k, b_v))

        if INPLACE_UPDATE and USE_INITIAL_STATE:
            index = tl.load(initial_state_indices + i_n).to(tl.int32)
            ht = initial_state + (index * stride_h + i_h * V * K).to(tl.int64)
            p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_ht2 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_ht3 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_ht4 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
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

    # GCU hardware limits
    major, _ = torch.gcu.get_device_capability("gcu")
    if major == 3:
        GCU_NUM_GRID = 24
    else:
        GCU_NUM_GRID = 48

    nv = triton.cdiv(V, 32)
    batch = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    total_work_items = int(batch) * H * nv
    grid = (min(total_work_items, GCU_NUM_GRID),)

    _chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu[grid](
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
        num_warps=2,
        num_stages=2,
        NV=nv,
        NUM_SPC=GCU_NUM_GRID,
        total_work_items=total_work_items,
    )
    return h, v_new


def patch_chunk_delta_h():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _CHUNK_DELTA_H_MODULE = "sglang.srt.layers.attention.fla.chunk_delta_h"
    HookRegistry.register(
        f"{_CHUNK_DELTA_H_MODULE}.chunk_gated_delta_rule_fwd_h",
        chunk_gated_delta_rule_fwd_h,
        HookType.REPLACE,
    )