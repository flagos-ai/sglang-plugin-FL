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

# Reference PyTorch implementation of chunk_gated_delta_rule (prefill / extend).

from __future__ import annotations

from typing import Optional

import torch


def chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Prefill/extend Gated-DeltaNet in pure PyTorch.

    Matches the recurrent update used by the packed-decode Triton kernel, run
    token-by-token per sequence. Updates ``initial_state`` in place when
    ``initial_state_indices`` is provided (SGLang CUDA-compat contract).

    Shapes (``head_first=False``):
      q/k: [B, T, H, K], v: [B, T, HV, V], g/beta: [B, T, HV]
      initial_state: [N, HV, V, K]

    Returns:
      (o, None, None) with o shaped like ``v``.
    """
    if head_first:
        raise NotImplementedError("head_first=True is not supported in the reference impl")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Flatten varlen: expected B=1 with cu_seqlens, or equal-length B sequences.
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"cu_seqlens requires batch size 1, got B={q.shape[0]}"
            )
        cu = cu_seqlens
        num_seqs = int(cu.numel()) - 1
        seq_ranges = [(int(cu[i]), int(cu[i + 1])) for i in range(num_seqs)]
        # Work on [T, H, ...] views
        q_ = q[0]
        k_ = k[0]
        v_ = v[0]
        g_ = g[0]
        beta_ = beta[0]
        batch_is_one = True
    else:
        B, T = q.shape[0], q.shape[1]
        num_seqs = B
        seq_ranges = [(b * T, (b + 1) * T) for b in range(B)]
        # Merge batch into time for a uniform loop, then unmerge.
        q_ = q.reshape(B * T, *q.shape[2:])
        k_ = k.reshape(B * T, *k.shape[2:])
        v_ = v.reshape(B * T, *v.shape[2:])
        g_ = g.reshape(B * T, *g.shape[2:])
        beta_ = beta.reshape(B * T, *beta.shape[2:])
        batch_is_one = False

    H = q_.shape[1]
    HV, V = v_.shape[1], v_.shape[2]
    K = k_.shape[-1]
    if HV % H != 0:
        raise ValueError(f"Invalid GQA config: H={H}, HV={HV}")
    G = HV // H

    o_ = torch.empty_like(v_)

    for seq_i, (start, end) in enumerate(seq_ranges):
        if initial_state is None:
            h = torch.zeros(HV, V, K, device=q.device, dtype=torch.float32)
            state_idx = None
        else:
            if initial_state_indices is not None:
                state_idx = int(initial_state_indices[seq_i])
            else:
                state_idx = seq_i
            h = initial_state[state_idx].float()  # [HV, V, K]

        for t in range(start, end):
            qt = q_[t].float()  # [H, K]
            kt = k_[t].float()  # [H, K]
            vt = v_[t].float()  # [HV, V]
            gt = g_[t].float()  # [HV]
            bt = beta_[t].float()  # [HV]

            if use_qk_l2norm_in_kernel:
                qt = qt * torch.rsqrt((qt * qt).sum(dim=-1, keepdim=True) + 1e-6)
                kt = kt * torch.rsqrt((kt * kt).sum(dim=-1, keepdim=True) + 1e-6)
            qt = qt * scale

            # Broadcast Q/K heads to HV (GQA).
            if G != 1:
                qt = qt.unsqueeze(1).expand(H, G, K).reshape(HV, K)
                kt = kt.unsqueeze(1).expand(H, G, K).reshape(HV, K)

            # h *= exp(g);  v -= h@k;  v *= beta;  h += outer(v,k);  o = h@q
            h = h * torch.exp(gt).view(HV, 1, 1)
            hk = (h * kt.view(HV, 1, K)).sum(dim=-1)  # [HV, V]
            vt = (vt - hk) * bt.unsqueeze(-1)
            h = h + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            o_[t] = (h * qt.view(HV, 1, K)).sum(dim=-1).to(dtype=v_.dtype)

        if initial_state is not None and state_idx is not None:
            initial_state[state_idx].copy_(h.to(dtype=initial_state.dtype))

    if batch_is_one:
        o = o_.unsqueeze(0)
    else:
        o = o_.reshape(q.shape[0], q.shape[1], HV, V)

    return o, None, None
