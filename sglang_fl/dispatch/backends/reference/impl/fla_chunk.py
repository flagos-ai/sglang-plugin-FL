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

"""Reference (PyTorch) implementation of chunk_gated_delta_rule.

Chunkwise extended-WY form (chunk_size=64) that is mathematically equivalent to
the token-by-token recurrence / Triton kernels, but avoids a Python loop over
every token — critical for prefill TTFT.

State layout matches SGLang's native FLA path: ``[N, H, V, K]`` (V-first).
When ``initial_state`` + ``initial_state_indices`` are provided, the cache is
updated in place (same contract as the CUDA ``INPLACE_UPDATE`` path).

CUDA-graph notes:
  * Equal-length batches (``cu_seqlens is None``) use fixed tensor shapes only.
  * Varlen ``cu_seqlens`` is resolved once on CPU (prefill; not graph-captured).
  * No per-token ``.item()`` / data-dependent host sync inside the compute path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def _expand_qk_to_v_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    """Expand q/k heads to match v heads for GVA (grouped value attention)."""
    h_qk = x.shape[-2]
    if h_qk == num_v_heads:
        return x
    if num_v_heads % h_qk != 0:
        raise ValueError(f"Invalid grouped heads: Hqk={h_qk}, Hv={num_v_heads}.")
    return x.repeat_interleave(num_v_heads // h_qk, dim=-2)


def _torch_chunkwise_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise gated-delta rule for a contiguous equal-length batch.

    Args:
        query/key: ``[B, T, H, K]``
        value: ``[B, T, H, V]``
        g/beta: ``[B, T, H]`` (``g`` in log space)
        scale: attention scale applied to ``q``
        initial_state: ``[B, H, V, K]`` or ``None``
        use_qk_l2norm_in_kernel: L2-normalize q/k along the last dim
        chunk_size: WY chunk length (default 64, matches FLA Triton)

    Returns:
        ``(output [B, T, H, V], final_state [B, H, V, K])``
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2, dim=-1, eps=1e-6)

    # Head-first float32: [B, H, T, ...]
    query = query.transpose(1, 2).contiguous().float()
    key = key.transpose(1, 2).contiguous().float()
    value = value.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    g = g.transpose(1, 2).contiguous().float()

    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * scale

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    padded_len = seq_len + pad_size
    n_chunks = padded_len // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, n_chunks, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)

    # Intra-chunk (I - tril(diag(beta) K K^T * decay))^{-1} via extended WY.
    tril_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(tril_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # Working state is K-first ``[B, H, K, V]`` (FLA chunkwise convention).
    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            k_dim,
            v_dim,
            device=value.device,
            dtype=torch.float32,
        )
    else:
        # Public API is V-first ``[B, H, V, K]``.
        state = (
            initial_state.to(device=value.device, dtype=torch.float32)
            .transpose(-1, -2)
            .contiguous()
        )

    out = torch.zeros_like(value)
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # Fixed iteration count = n_chunks (static for a given capture T).
    for i in range(n_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        decay_i = decay_mask[:, :, i]
        g_i = g[:, :, i]

        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_i).masked_fill(causal_mask, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ state
        out[:, :, i] = attn_inter + attn_i @ v_new
        state = state * g_i[:, :, -1, None, None].exp() + (
            k_i * (g_i[:, :, -1, None] - g_i).exp().unsqueeze(-1)
        ).transpose(-1, -2) @ v_new

    out = out.reshape(batch_size, num_heads, padded_len, v_dim)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    # Back to V-first ``[B, H, V, K]``.
    final_state = state.transpose(-1, -2).contiguous()
    return out, final_state


def chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    Reference implementation of ``chunk_gated_delta_rule``.

    Signature matches ``sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule``
    / the FL bridge. Returns ``(o, None, h)``; ``h`` is always ``None`` here because
    the recurrent path updates ``initial_state`` in place when a cache pool is given.
    """
    if head_first:
        raise NotImplementedError(
            "head_first=True is deprecated; please pass head_first=False "
            "with inputs shaped [B, T, H, ...]."
        )

    assert q.dtype == k.dtype == v.dtype
    assert len(beta.shape) == 3, (
        "beta must be of shape [B, T, H] if head_first=False."
    )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    h_v = v.shape[2]
    v_dim = v.shape[-1]
    k_dim = k.shape[-1]
    device = q.device

    # ---- equal-length batch: one chunkwise call, graph-friendly ----
    if cu_seqlens is None:
        num_seqs = q.shape[0]
        q_e = _expand_qk_to_v_heads(q, h_v)
        k_e = _expand_qk_to_v_heads(k, h_v)

        idx = None
        if initial_state is not None and initial_state_indices is not None:
            idx = initial_state_indices.long()
            states = initial_state.index_select(0, idx).to(torch.float32)
        elif initial_state is not None:
            states = initial_state.to(torch.float32)
            if states.shape[0] != num_seqs:
                raise ValueError(
                    f"initial_state batch dim ({states.shape[0]}) must equal "
                    f"number of sequences ({num_seqs}) when indices are not given."
                )
        else:
            states = None

        out, final_states = _torch_chunkwise_gated_delta_rule(
            query=q_e,
            key=k_e,
            value=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=states,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        if initial_state is not None:
            final = final_states.to(dtype=initial_state.dtype)
            if idx is not None:
                initial_state.index_copy_(0, idx, final)
            else:
                initial_state.copy_(final)

        return out, None, None

    # ---- varlen (prefill): resolve ranges once on CPU, no per-token sync ----
    if q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} "
            f"when using `cu_seqlens`."
        )

    # Host list once — not inside a CUDA graph capture (varlen prefill).
    cu_list = cu_seqlens.detach().to("cpu").tolist()
    num_seqs = len(cu_list) - 1
    if (
        initial_state_indices is not None
        and initial_state_indices.shape[0] != num_seqs
    ):
        raise ValueError(
            f"The number of initial states is expected to be equal to the "
            f"number of input sequences, i.e., {num_seqs} rather "
            f"than {initial_state_indices.shape[0]}."
        )

    idx = None
    if initial_state is not None and initial_state_indices is not None:
        idx = initial_state_indices.long()
        states = initial_state.index_select(0, idx).to(torch.float32)
    elif initial_state is not None:
        states = initial_state.to(torch.float32)
        if states.shape[0] != num_seqs:
            raise ValueError(
                f"initial_state batch dim ({states.shape[0]}) must equal "
                f"number of sequences ({num_seqs}) when indices are not given."
            )
    else:
        states = torch.zeros(
            num_seqs, h_v, v_dim, k_dim, dtype=torch.float32, device=device
        )

    out = torch.zeros_like(v)

    for seq_i in range(num_seqs):
        start = int(cu_list[seq_i])
        end = int(cu_list[seq_i + 1])
        if end <= start:
            continue

        q_seq = _expand_qk_to_v_heads(q[0, start:end], h_v).unsqueeze(0)
        k_seq = _expand_qk_to_v_heads(k[0, start:end], h_v).unsqueeze(0)
        v_seq = v[0, start:end].unsqueeze(0)
        g_seq = g[0, start:end].unsqueeze(0)
        beta_seq = beta[0, start:end].unsqueeze(0)

        out_seq, final_state = _torch_chunkwise_gated_delta_rule(
            query=q_seq,
            key=k_seq,
            value=v_seq,
            g=g_seq,
            beta=beta_seq,
            scale=scale,
            initial_state=states[seq_i : seq_i + 1],
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        out[0, start:end] = out_seq[0]
        states[seq_i] = final_state[0]

    if initial_state is not None:
        final = states.to(dtype=initial_state.dtype)
        if idx is not None:
            initial_state.index_copy_(0, idx, final)
        else:
            initial_state.copy_(final)

    return out, None, None
