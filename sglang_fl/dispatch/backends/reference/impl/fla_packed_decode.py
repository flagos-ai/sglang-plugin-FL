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

"""Reference (PyTorch) implementation of packed-decode Gated-DeltaNet.

Mirrors ``fused_recurrent_gated_delta_rule_packed_decode_kernel``:
unpack mixed QKV, apply fused gating (softplus + sigmoid), then a T=1
recurrent update. State layout is SGLang native ``[N, HV, V, K]``.
Cache rows selected by ``ssm_state_indices`` are updated in place.

CUDA-graph safe: fixed shapes, no host sync (``.item()`` / ``.tolist()`` /
``bool(tensor)``), invalid indices handled via mask + ``clamp`` +
``index_copy_`` of full ``[B, ...]`` rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Matches fused_recurrent_gated_delta_rule_packed_decode_kernel.
_SOFTPLUS_THRESHOLD = 20.0


def _index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """ATen index_select — bypasses flag_gems Tensor.__getitem__ patches."""
    return torch.ops.aten.index_select(x, dim, index.contiguous())


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
    """
    Packed-QKV decode Gated-DeltaNet in PyTorch.

    Args:
        mixed_qkv: ``[B, H*K + H*K + HV*V]`` (a stray token axis is flattened)
        a, b: ``[B, HV]``
        A_log, dt_bias: ``[HV]``
        scale: q scale
        initial_state: cache pool ``[N, HV, V, K]``
        out: ``[B, 1, HV, V]`` (or any view with the same numel)
        ssm_state_indices: ``[B]``; negative indices skip the update and write zeros
        use_qk_l2norm_in_kernel: L2-normalize q/k along the last dim

    Returns:
        ``(out, initial_state)`` with both tensors updated in place.
    """
    if mixed_qkv.ndim == 3:
        mixed_qkv = mixed_qkv.reshape(mixed_qkv.shape[0], -1)
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )

    B = mixed_qkv.shape[0]
    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    HV, V, K = initial_state.shape[-3:]

    qkv_dim = mixed_qkv.shape[1]
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

    q = mixed_qkv[:, : H * K].reshape(B, H, K).float()
    k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, H, K).float()
    v = mixed_qkv[:, 2 * H * K : 2 * H * K + HV * V].reshape(B, HV, V).float()
    if HV != H:
        group = HV // H
        q = q.repeat_interleave(group, dim=1)
        k = k.repeat_interleave(group, dim=1)

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
    q = q * scale

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(x <= _SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
    g = -A_log.float().exp() * softplus_x
    beta = torch.sigmoid(b.float())

    idx = ssm_state_indices.long()
    valid = idx >= 0
    safe_idx = idx.clamp(min=0)
    # Always mask — no bool(valid.all()) host sync (breaks CUDA/HIP graph capture).
    state = _index_select(initial_state, 0, safe_idx).to(torch.float32)
    state = state.masked_fill(~valid[:, None, None, None], 0)

    # state: [B, HV, V, K]; q/k: [B, HV, K]; v: [B, HV, V]
    state = state * g[:, :, None, None].exp()
    kv_mem = (state * k[:, :, None, :]).sum(dim=-1)
    delta = (v - kv_mem) * beta[:, :, None]
    state = state + delta[:, :, :, None] * k[:, :, None, :]
    o = (state * q[:, :, None, :]).sum(dim=-1)
    o = o.masked_fill(~valid[:, None, None], 0)

    out.copy_(o.to(dtype=out.dtype).reshape(out.shape))

    # In-place cache writeback. Fixed Python loop over B (static in CUDA graph).
    # Invalid rows write the *current* target row (true no-op), so a -1 that
    # clamps onto the same slot as a valid row cannot clobber that update —
    # bulk index_copy_ with duplicate indices is non-deterministic under graph replay.
    final = state.to(dtype=initial_state.dtype)
    for b in range(B):
        tgt = safe_idx[b : b + 1]
        cur = _index_select(initial_state, 0, tgt)
        new_row = torch.where(valid[b], final[b].unsqueeze(0), cur)
        initial_state.index_copy_(0, tgt, new_row)

    return out, initial_state
