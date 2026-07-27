# Gcu FLA (Flash Linear Attention) operator implementations.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

def chunk_gated_delta_rule_gcu(
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
    from sglang.srt.layers.attention.fla.chunk import ChunkGatedDeltaRuleFunction

    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    return o, None, h

def fused_recurrent_gated_delta_rule_gcu(
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

def fused_recurrent_gated_delta_rule_packed_decode_gcu(
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
):
    # During CUDA Graph capture, SGLang may pass out as (B, 1, HV, V) instead of
    # (B, HV, V). Squeeze to canonical shape — contiguous so this is a free view.
    _orig_out = out

    if out.ndim == 4:
        if out.shape[1] != 1:
            raise ValueError(
                f"`out` as 4D must have shape (B, 1, HV, V) "
                f"(got out.shape={tuple(out.shape)})."
            )
        out = out.squeeze(1)

    """GCU-compatible packed decode: uses 1-D grid to avoid grid.y > 255."""
    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim}).")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim}).")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if any(t.device != dev for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim}).")
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements."
        )
    if out.shape != (B, HV, V):
        raise ValueError(f"`out` must have shape {(B, HV, V)} (got out.shape={tuple(out.shape)}).")

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}.")
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}.")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK}).")
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1

    NV = triton.cdiv(V, BV)

    # GCU hardware limits
    major, _ = torch.gcu.get_device_capability("gcu")
    if major == 3:
        GCU_NUM_GRID = 24
        GCU_MAX_DSM_MEMORY = int(1.5 * 1024 * 1024)  # 1536 KB
    else:
        GCU_NUM_GRID = 48
        GCU_MAX_DSM_MEMORY = 917504 // 2  # 448 KB

    dsm_peak = BV * BK * 4
    if dsm_peak > GCU_MAX_DSM_MEMORY:
        raise ValueError(
            f"DSM peak {dsm_peak} B exceeds hardware limit {GCU_MAX_DSM_MEMORY} B "
            f"(BV={BV}, BK={BK}); reduce V or K."
        )

    # GCU: convert int64 ssm_state_indices to int32
    if ssm_state_indices.dtype == torch.int64:
        ssm_state_indices = ssm_state_indices.to(torch.int32)

    total_work = NV * B * HV
    grid = (GCU_NUM_GRID,)

    _gcu_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        total_work=total_work,
        H=H,
        HV=HV,
        K=K,
        V=V,
        NV=NV,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NUM_SPC=GCU_NUM_GRID,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state

# ---------------------------------------------------------------------------
# GCU Packed Decode Triton kernel (1-D grid, avoids grid.y > 255)
# ---------------------------------------------------------------------------

@triton.jit
def _gcu_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    total_work,  # gcu patch: total work items = NV * B * HV
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NV: tl.constexpr,  # gcu patch: number of V tiles
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NUM_SPC: tl.constexpr,  # gcu patch: number of grid blocks
):
    """
    GCU Fixed Grid: single kernel launch processes all (B, HV, NV) work items.

    Each work item maps to original 2-D grid (i_v, i_nh):
      work_id = i_nh * NV + i_v
      i_v   = work_id % NV        (V tile index)
      i_nh  = work_id // NV       (= i_n * HV + i_hv)
      i_n   = i_nh // HV          (batch index)
      i_hv  = i_nh % HV           (V-head index)
      i_h   = i_hv // (HV // H)   (Q/K head index)
    """
    LOG2E: tl.constexpr = 1.4426950408889634  # log₂(e)

    pid = tl.program_id(0)

    for work_id in range(pid, total_work, NUM_SPC):
        i_v = work_id % NV
        i_nh = work_id // NV
        i_n = i_nh // HV
        i_hv = i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int32)

        p_o = o + (i_n * HV + i_hv) * V + o_v

        if state_idx < 0:
            zero = tl.zeros([BV], dtype=tl.float32)
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
        else:
            # Load SSM state slice b_h: [BV, BK]
            h_base = state_idx.to(tl.int64) * stride_init_state_token
            p_h0 = (
                h0 + h_base + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            )
            b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

            # Load Q / K / V
            p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
            b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            b_k = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0.0).to(tl.float32)

            # Optional L2 norm
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.math.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.math.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale

            # Load gating params
            a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
            b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
            A_log_val = tl.load(A_log + i_hv).to(tl.float32)
            dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

            # softplus + decay factor (exp → exp2 for math optimization)
            x = a_val + dt_bias_val
            exp_x = tl.math.exp2(x * LOG2E)
            softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + exp_x), x)
            g_val = -tl.math.exp2(A_log_val * LOG2E) * softplus_x
            beta_val = 1.0 / (1.0 + tl.math.exp2(-b_val * LOG2E))

            # Gated Delta Rule update
            exp_g = tl.math.exp2(g_val * LOG2E)
            b_h = b_h * exp_g                                    # decay
            b_v = b_v - tl.sum(b_h * b_k[None, :], 1)           # delta residual
            b_v = b_v * beta_val                                  # gating
            b_h = b_h + b_v[:, None] * b_k[None, :]             # rank-1 update
            b_o = tl.sum(b_h * b_q[None, :], 1)                 # output projection

            # Write output
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # In-place state update
            p_ht = ht + h_base + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)