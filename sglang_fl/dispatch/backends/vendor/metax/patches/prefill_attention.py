# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip, is_maca

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_maca = is_maca()

if _is_cuda or _is_hip:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk  # mask the padded tail of the D dimension

    off_q = (
        (cur_batch_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :],
        other=0.0,
    )

    k_base = K + cur_kv_head * stride_kh
    v_base = V + cur_kv_head * stride_vh

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Causal: a query only attends to keys up to its own position.
    end_n = (
        tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
        if IS_CAUSAL
        else cur_batch_seq_len
    )

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < cur_batch_seq_len

        off_k = (cur_batch_start + n_idx[None, :]) * stride_kbs + offs_d[:, None]
        k = tl.load(k_base + off_k, mask=n_mask[None, :] & mask_d[:, None], other=0.0)

        qk = tl.dot(q, k) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(
                n_mask[None, :] & (offs_m[:, None] >= n_idx[None, :]),
                qk,
                float("-inf"),
            )
        else:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        # ---- FlashAttention-2 online softmax update ----
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        off_v = (cur_batch_start + n_idx[:, None]) * stride_vbs + offs_d[None, :]
        v = tl.load(v_base + off_v, mask=n_mask[:, None] & mask_d[None, :], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    # Normalize once, at the end.
    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + off_o,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :],
    )


def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, sm_scale=None
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    sm_scale: softmax scale, defaults to 1/sqrt(head_dim)
    """
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    if _is_maca and not is_causal:
        # C500X-tuned for non-causal (vision encoder) attention. Validated for
        # head_dim <= 128 (BLOCK_DMODEL <= 128), where 128/32/8/2 fits exactly in
        # the 64 KB shared-memory limit. Larger head_dim would not fit -> shrink.
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 32, 8, 1
        if BLOCK_DMODEL > 128:
            BLOCK_M, BLOCK_N, num_stages = 64, 32, 1
    else:
        if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:
            BLOCK_M = BLOCK_N = 128
        else:
            BLOCK_M = BLOCK_N = 64
        num_warps = 4 if Lk <= 64 else 8
        num_stages = 1

    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
    num_stages = 1
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
    )
    return o


def patch_prefill_attention() -> None:
    """Replace SGLang's prefill attention with the MetaX-compatible kernel."""
    import importlib

    module_name = "sglang.srt.layers.attention.triton_ops.prefill_attention"
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(f"failed to import {module_name}") from error

    module.context_attention_fwd = context_attention_fwd
