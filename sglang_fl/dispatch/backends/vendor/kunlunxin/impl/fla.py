# Copyright 2026 FlagOS Contributors
#
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

# Kunlunxin (Baidu Kunlun XPU) FLA (Flash Linear Attention) operator implementations.
#
# Both the prefill (chunk) and decode (packed) Gated-DeltaNet paths are backed by
# the single ``sgl_kernel`` chunk op ``torch.ops.sgl_kernel.klx_gated_delta_net``:
#   chunk_gated_delta_rule (prefill)        -> klx_gated_delta_net
#   fused_recurrent_gated_delta_rule_packed_decode (decode, treated as a T=1 chunk)
#                                           -> klx_gated_delta_net
#
# klx_gated_delta_net schema (positional):
#   (q, k, v, g, beta, scale, initial_state, output_final_state: bool,
#    cu_seqlens, head_first: bool, use_qk_l2norm_in_kernel: bool) -> Tensor[]
# returning [output, final_state, ...]. q/k are [1, T, H, K], v is [1, T, HV, V],
# g/beta are [1, T, HV]; cu_seqlens MUST be a CPU int32 tensor. q/k are L2-normalised
# and q is scaled internally when use_qk_l2norm_in_kernel=True.
#
# State layout: SGLang stores ssm_states as [.., HV, V, K]; klx_gated_delta_net
# expects initial_state as [N, HV, K, V] (last two dims swapped), so the state is
# transposed going in and the returned final state is transposed coming back out.
# (Verified numerically to match the previous xtorch_ops path exactly: prefill
# out/final-state diff 0.0, decode out diff ~6e-5, state diff ~1.5e-3.)

from __future__ import annotations

from typing import Optional

import torch

# Softplus parameters used by SGLang's Gated-DeltaNet gating (see
# fused_recurrent_gated_delta_rule_packed_decode_kernel): softplus_beta=1.0,
# threshold=20.0 (SOFTPLUS_THRESHOLD).
_SOFTPLUS_THRESHOLD = 20.0


def _fla_chunk_size() -> int:
    """The chunk granularity of the klx chunk kernel (FLA CHUNK_SIZE)."""
    try:
        from sglang.kernels.ops.attention.fla.chunk_delta_h import (
            CHUNK_SIZE as _FLA_CHUNK_SIZE,
        )

        return int(_FLA_CHUNK_SIZE)
    except Exception:
        return 64


def _mamba_cache_chunk_size(fallback: int) -> int:
    """SGLang's mamba state-checkpoint granularity (max(chunk, page_size))."""
    try:
        from sglang.srt.runtime_context import mamba_cache_chunk_size

        return int(mamba_cache_chunk_size())
    except Exception:
        return fallback


def _select_state_checkpoints(
    hk: torch.Tensor, cu_seqlens: torch.Tensor, state_dtype: torch.dtype
) -> torch.Tensor:
    """Turn the kernel's per-chunk states into SGLang's checkpoint tensor.

    ``hk`` is ``[sum(ceil(L_i / fla_chunk)), HV, K, V]``: request i owns a block
    of ``ceil(L_i / fla_chunk)`` rows whose row 0 is the *incoming* state and
    row j is the state after j chunks (the state after the final chunk is
    returned separately as ``final_state``).  SGLang's
    ``_init_track_ssm_indices`` walks a flat list where request i owns
    ``ceil(L_i / cache_chunk)`` entries indexed from its block start, and reads
    the state after ``L_i // cache_chunk`` chunks for an unaligned prefix.

    Returns ``[1, total_checkpoints, HV, V, K]`` (SGLang's V,K state layout).
    """
    fla_chunk = _fla_chunk_size()
    cache_chunk = max(fla_chunk, _mamba_cache_chunk_size(fla_chunk))
    step = max(1, cache_chunk // fla_chunk)

    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    keep = []
    row = 0
    for length in lens:
        num_chunks = (int(length) + fla_chunk - 1) // fla_chunk
        num_cache_states = (int(length) + cache_chunk - 1) // cache_chunk
        for k in range(1, num_cache_states + 1):
            # Clamp to the last available row: the read index SGLang uses for an
            # unaligned prefix (floor(L / chunk)) is always within the block.
            keep.append(row + min(k * step, num_chunks - 1))
        row += num_chunks
    if not keep:
        return hk.new_empty((1, 0) + hk.shape[1:]).transpose(-1, -2)
    idx = torch.tensor(keep, dtype=torch.long, device=hk.device)
    # [N, HV, K, V] -> [1, N, HV, V, K]
    return hk.index_select(0, idx).transpose(-1, -2).unsqueeze(0).to(state_dtype)


def chunk_gated_delta_rule_kunlunxin(
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
    """Prefill-path Gated-DeltaNet on Kunlunxin Platform.

    Routed through ``torch.ops.sgl_kernel.klx_gated_delta_net`` (klx chunk kernel).

    On the CUDA path (Kunlun is cuda-compat) SGLang passes ``initial_state`` as
    the FULL ssm_states cache plus ``initial_state_indices`` selecting the rows
    for this batch, and does NOT read back the returned final state — the cache
    must be updated in place. So we gather the per-request states, run the
    kernel, and scatter the final states back into ``initial_state``.

    State layout: SGLang stores ssm_states as [.., HV, V, K]; the klx kernel
    expects ``initial_state`` as (N, HV, K, V), so the last two dims are swapped
    going in and coming back out.

    Returns (o, None, None) to match the SGLang chunk_gated_delta_rule contract
    (the CUDA caller ignores the recurrent-state return value).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Gate/beta dtype must track the activation dtype.
    if q.dtype == torch.float16:
        g = g.contiguous().half()
        beta = beta.contiguous().half()
    else:
        g = g.contiguous().float()
        beta = beta.contiguous().float()

    # cu_seqlens must be an int32 CPU tensor for the klx chunk kernel.
    cu = cu_seqlens.to(device="cpu", dtype=torch.int32).contiguous()
    N = cu.numel() - 1

    # The kernel rejects mixed q/initial_state dtypes: match the activation
    # dtype (fp16 -> fp16 state, otherwise fp32).
    state_dtype = torch.float16 if q.dtype == torch.float16 else torch.float32

    # index_select/index_copy_ require a long index tensor; cache_indices is int32.
    idx = initial_state_indices.long() if initial_state_indices is not None else None

    # Gather per-request incoming states out of the full ssm_states cache and
    # transpose SGLang's [N, HV, V, K] into the kernel's [N, HV, K, V].
    if initial_state is None:
        HV, V = v.shape[2], v.shape[-1]
        K = k.shape[-1]
        h0 = k.new_zeros(N, HV, K, V, dtype=state_dtype)
    else:
        if idx is not None:
            gathered = initial_state.index_select(0, idx)
        else:
            gathered = initial_state
        # [N, HV, V, K] -> [N, HV, K, V]
        h0 = gathered.transpose(-1, -2).contiguous().to(state_dtype)

    res = torch.ops.sgl_kernel.klx_gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        scale,
        h0,
        True,  # output_final_state
        cu,
        head_first,
        use_qk_l2norm_in_kernel,
    )
    o, ht = res[0], res[1]

    # Scatter the final states back into the cache, undoing the transpose.
    if initial_state is not None and idx is not None:
        final = ht.transpose(-1, -2).to(initial_state.dtype)
        initial_state.index_copy_(0, idx, final)

    # The klx chunk kernel also returns the state after every chunk
    # (res[2] = [sum(ceil(L_i / chunk) + 1), HV, K, V], row 0 being the incoming
    # state).  SGLang's radix-cache mamba tracking consumes those as `h`; without
    # them `_track_mamba_state_extend` trips `assert h is not None` whenever a
    # prefix is not chunk-aligned.
    if len(res) > 2 and res[2] is not None:
        h = _select_state_checkpoints(res[2], cu, state_dtype)
    else:
        h = None

    return o, None, h


def fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
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
    """Packed-QKV decode Gated-DeltaNet on Kunlunxin Platform.

    mixed_qkv [B, qkv_dim] (2D, decode T=1), a/b [B, HV], A_log/dt_bias [HV],
    initial_state [num_states, HV, V, K], out [B, 1, HV, V], ssm_state_indices [B].
    State is updated in place; returns (out, initial_state).

    sgl_kernel exposes no dedicated recurrent decode op, so decode is run as a
    single-token (T=1) chunk through ``klx_gated_delta_net``. This means we must
    reproduce, in torch, the QKV split + gating that the fused packed-decode
    kernel performs internally:
      q, k, v = split(mixed_qkv)                          (q/k L2-normed by klx)
      g       = -exp(A_log) * softplus(a + dt_bias)       (softplus_beta=1, threshold=20)
      beta    = sigmoid(b)
    then gather/transpose the per-request states, run the chunk kernel with one
    token per sequence, and scatter the final states back.
    """
    # SGLang passes mixed_qkv as 2D [B, qkv_dim]; flatten any stray token axis.
    if mixed_qkv.ndim == 3:
        mixed_qkv = mixed_qkv.reshape(mixed_qkv.shape[0], -1)

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[-1]
    q_dim = (qkv_dim - HV * V) // 2
    H = q_dim // K

    # Split the packed projection: [q (H*K) | k (H*K) | v (HV*V)].
    qf = mixed_qkv[:, : H * K].reshape(B, 1, H, K).contiguous()
    kf = mixed_qkv[:, H * K : 2 * H * K].reshape(B, 1, H, K).contiguous()
    vf = mixed_qkv[:, 2 * H * K : 2 * H * K + HV * V].reshape(B, 1, HV, V).contiguous()

    # Gating (matches the fused packed-decode kernel exactly).
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(x <= _SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
    g = (-A_log.float().exp() * softplus_x).reshape(B, 1, HV)
    beta = torch.sigmoid(b.float()).reshape(B, 1, HV)

    state_dtype = torch.float16 if qf.dtype == torch.float16 else torch.float32
    # klx_gated_delta_net requires g/beta to be fp16 or fp32 (bf16 is rejected);
    # keep them in the state dtype (fp16 for fp16 activations, else fp32).
    g = g.to(state_dtype).contiguous()
    beta = beta.to(state_dtype).contiguous()

    # One token per sequence -> cu_seqlens [0, 1, 2, ..., B] on CPU int32.
    cu = torch.arange(0, B + 1, dtype=torch.int32)

    idx = ssm_state_indices.long()
    gathered = initial_state.index_select(0, idx)  # [B, HV, V, K]
    h0 = gathered.transpose(-1, -2).contiguous().to(state_dtype)  # [B, HV, K, V]

    res = torch.ops.sgl_kernel.klx_gated_delta_net(
        qf,
        kf,
        vf,
        g,
        beta,
        scale,
        h0,
        True,  # output_final_state
        cu,
        False,  # head_first
        use_qk_l2norm_in_kernel,
    )
    o, ht = res[0], res[1]

    out.copy_(o.reshape(out.shape))

    # Scatter updated states back into the cache, undoing the transpose.
    final = ht.transpose(-1, -2).to(initial_state.dtype)
    initial_state.index_copy_(0, idx, final)

    return out, initial_state
