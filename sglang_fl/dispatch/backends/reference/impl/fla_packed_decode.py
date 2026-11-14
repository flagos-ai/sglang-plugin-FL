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

# Reference PyTorch implementation of fused_recurrent_gated_delta_rule_packed_decode.
#
# Numerically matches SGLang's Triton packed-decode kernel
# (sglang.srt.layers.attention.fla.fused_recurrent):
#   g = -exp(A_log) * softplus(a + dt_bias)   # softplus_beta=1, threshold=20
#   beta = sigmoid(b)
#   h *= exp(g)
#   v -= h @ k;  v *= beta
#   h += outer(v, k)
#   o = h @ q
#
# Designed for CUDA-graph capture:
#   - fixed tensor shapes, no Python data-dependent control flow
#   - no .item() / CPU sync
#   - in-place updates to ``out`` and ``initial_state``

from __future__ import annotations

import torch

_SOFTPLUS_THRESHOLD = 20.0


def fused_recurrent_gated_delta_rule_packed_decode_torch(
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
    """Packed-QKV Gated-DeltaNet decode (PyTorch reference).

    Args:
        mixed_qkv: [B, qkv_dim] packed projection, layout [q | k | v].
        a, b: [B, HV] gating inputs.
        A_log, dt_bias: [HV].
        scale: attention scale (typically K ** -0.5).
        initial_state: [num_slots, HV, V, K] state pool (updated in-place).
        out: [B, 1, HV, V] preallocated output buffer (written in-place).
        ssm_state_indices: [B] per-request slot indices; negative => pad
            (zero output, state untouched).
        use_qk_l2norm_in_kernel: L2-normalise q/k inside the op.

    Returns:
        (out, initial_state)
    """
    if mixed_qkv.ndim == 3:
        mixed_qkv = mixed_qkv.reshape(mixed_qkv.shape[0], -1)
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]

    qkv_dim = mixed_qkv.shape[-1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    G = HV // H

    # Split packed QKV and broadcast Q/K heads to HV (GQA).
    q = mixed_qkv[:, : H * K].reshape(B, H, K)
    k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, H, K)
    v = mixed_qkv[:, 2 * H * K : 2 * H * K + HV * V].reshape(B, HV, V)

    # fp32 accumulate for numerical parity with the Triton kernel.
    q = q.float()
    k = k.float()
    v = v.float()

    if G != 1:
        q = q.unsqueeze(2).expand(B, H, G, K).reshape(B, HV, K)
        k = k.unsqueeze(2).expand(B, H, G, K).reshape(B, HV, K)

    if use_qk_l2norm_in_kernel:
        # Match Triton: q / sqrt(sum(q^2) + 1e-6)
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * scale

    # Gating (softplus_beta=1.0, threshold=20.0).
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= _SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x
    )
    g = -A_log.float().exp() * softplus_x  # [B, HV]
    beta = torch.sigmoid(b.float())  # [B, HV]

    # CUDA-graph-safe gather: clamp negative pad indices; mask outputs later.
    idx = ssm_state_indices.to(dtype=torch.long)
    valid = idx >= 0
    safe_idx = idx.clamp(min=0)
    valid_bhvk = valid.view(B, 1, 1, 1)

    h = initial_state.index_select(0, safe_idx).float()  # [B, HV, V, K]
    h_orig = h

    # Recurrent gated-delta update (vectorised over B and HV).
    h = h * torch.exp(g).view(B, HV, 1, 1)
    h_k = torch.matmul(h, k.unsqueeze(-1)).squeeze(-1)  # [B, HV, V]
    v = (v - h_k) * beta.unsqueeze(-1)
    h = h + v.unsqueeze(-1) * k.unsqueeze(-2)
    o = torch.matmul(h, q.unsqueeze(-1)).squeeze(-1)  # [B, HV, V]

    # Pad (idx < 0): zero output; identity-write the gathered state back.
    o = o.masked_fill(~valid.view(B, 1, 1), 0)
    h_store = torch.where(valid_bhvk, h, h_orig).to(initial_state.dtype)

    out.view(B, HV, V).copy_(o.to(dtype=out.dtype))
    initial_state.index_copy_(0, safe_idx, h_store)

    return out, initial_state
