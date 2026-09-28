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

"""Kunlunxin OOT attention backend (PyTorch, no sgl_kernel)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class KunlunxinForwardMetadata:
    block_kv_indices: Optional[torch.Tensor] = None
    context_len_tensor: Optional[torch.Tensor] = None


def create_kv_indices_native(
    req_to_token,
    req_pool_indices,
    block_kv_indices,
    max_seqlen_pad: int,
    page_size: int,
):
    max_req_len = max_seqlen_pad * page_size
    cur_max_req_tokens = req_to_token[req_pool_indices, :max_req_len]
    block_indices = (cur_max_req_tokens[:, ::page_size]) // page_size
    block_kv_indices.copy_(block_indices)


def _as_paged_cache(cache: torch.Tensor, page_size: int, num_heads: int, head_dim: int):
    """View flat ``[tokens, heads, dim]`` KV as FA-style paged layout.

    Returns ``[num_pages, page_size, heads, dim]``.
    """
    if cache.ndim == 4:
        if cache.shape[1] == page_size:
            return cache
        if cache.shape[2] == page_size:
            return cache.transpose(1, 2).contiguous()
        return cache
    return cache.view(-1, page_size, num_heads, head_dim)


def _paged_attention_decode_torch(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    scale: float,
    num_kv_heads: int,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA-graph-safe paged attention decode (batched, no CPU sync).

    Replaces ``torch.ops.sgl_kernel.klx_attention_decode``.

    Single gather + masked attention (few CUDA launches → fast Graph capture).
    ``context_lens`` masks pad tokens / unused ``block_table`` entries (``-1``).
    """
    B, Hq, Dk = query.shape
    Dv = output.shape[-1]
    max_blocks = block_table.shape[1]
    max_seq = max_blocks * block_size

    k_paged = _as_paged_cache(k_cache, block_size, num_kv_heads, Dk)
    v_paged = _as_paged_cache(v_cache, block_size, num_kv_heads, Dv)

    safe_bt = block_table.clamp(min=0).to(dtype=torch.long)
    # [B, max_blocks, page_size, Hkv, D] -> [B, max_seq, Hkv, D]
    k = k_paged[safe_bt].reshape(B, max_seq, num_kv_heads, Dk)
    v = v_paged[safe_bt].reshape(B, max_seq, num_kv_heads, Dv)

    if Hq != num_kv_heads:
        if Hq % num_kv_heads != 0:
            raise ValueError(
                f"GQA requires Hq % Hkv == 0 (got Hq={Hq}, Hkv={num_kv_heads})."
            )
        rep = Hq // num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    # [B, H, S, D]
    q_h = query.unsqueeze(2).float()  # [B, Hq, 1, Dk]
    k_h = k.transpose(1, 2).float()  # [B, Hq, max_seq, Dk]
    v_h = v.transpose(1, 2).float()

    # Graph-safe pad mask without boolean `<` / `masked_fill`.
    # Kunlunxin flag_gems `lt` Triton fails on large int compares during
    # CUDA Graph capture; use float arange + clamp only (no host sync).
    # invalid==1 iff pos >= context_len  (same as ~(pos < context_lens)).
    pos = torch.arange(max_seq, device=query.device, dtype=torch.float32)
    cl = context_lens.to(dtype=torch.float32).unsqueeze(1)  # [B, 1]
    invalid = (pos.unsqueeze(0) - cl + 1.0).clamp(min=0.0, max=1.0)  # [B, max_seq]
    key_mask_f = 1.0 - invalid

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # [B, Hq, 1, S]
    scores = scores + (-1.0e9) * invalid[:, None, None, :]

    if sinks is not None:
        sink = sinks.float().view(1, Hq, 1, 1)
        row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        exp_scores = torch.exp(scores - row_max)
        exp_scores = exp_scores * key_mask_f[:, None, None, :].to(exp_scores.dtype)
        denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
        attn = exp_scores / denom
    else:
        attn = torch.softmax(scores, dim=-1)
        # Fully-masked rows → nan from softmax; zero them.
        attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.matmul(attn, v_h).squeeze(2)  # [B, Hq, Dv]
    output.copy_(out.to(dtype=output.dtype))
    return output


def _sdpa_forward_extend_torch(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    """Varlen extend via per-sequence SDPA (TorchNativeAttnBackend contract)."""
    if k_cache.ndim == 4:
        k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

    query = query.movedim(0, query.dim() - 2)
    start_q = 0
    for seq_idx in range(int(seq_lens.shape[0])):
        extend_seq_len_q = int(extend_seq_lens[seq_idx])
        prefill_seq_len_q = int(extend_prefix_lens[seq_idx])
        seq_len_kv = int(seq_lens[seq_idx])
        end_q = start_q + extend_seq_len_q

        per_req_query = query[:, start_q:end_q, :]
        per_req_query_redundant = torch.empty(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )
        per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query

        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
            per_req_key = per_req_key.to(dtype=per_req_query.dtype)
            per_req_value = per_req_value.to(dtype=per_req_query.dtype)

        per_req_out_redundant = (
            scaled_dot_product_attention(
                per_req_query_redundant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out_redundant[prefill_seq_len_q:, :, :]
        start_q = end_q
    return output


class KunlunxinBackend(AttentionBackend):
    """Kunlunxin full-attention backend for GDN-hybrid full layers (PyTorch)."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.forward_metadata: Optional[KunlunxinForwardMetadata] = None

        # Kept for API compatibility with the previous klx backend.
        self.cum_q_lod = None
        self.cum_q_lod_cpu = None
        self.cum_kv_lod = None
        self.cum_kv_lod_cpu = None
        self.max_seq_q = None
        self.max_seq_kv = None

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        bs = forward_batch.batch_size

        if forward_batch.forward_mode.is_decode_or_idle():
            adjusted_seq_lens = forward_batch.seq_lens
            adjusted_seq_lens_cpu = forward_batch.seq_lens_cpu
            max_seqlen_pad = (
                adjusted_seq_lens_cpu.max().item() + self.page_size - 1
            ) // self.page_size
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad), -1, dtype=torch.int32, device=self.device
            )
            create_kv_indices_native(
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_kv_indices,
                max_seqlen_pad,
                self.page_size,
            )
            self.forward_metadata = KunlunxinForwardMetadata(
                block_kv_indices=block_kv_indices,
                context_len_tensor=adjusted_seq_lens.int().contiguous(),
            )
        else:
            self.cum_q_lod = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cum_q_lod[1:] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            self.cum_q_lod_cpu = torch.zeros((bs + 1,), dtype=torch.int32, device="cpu")
            self.cum_q_lod_cpu[1:] = torch.cumsum(
                torch.tensor(forward_batch.extend_seq_lens_cpu), dim=0
            )
            self.cum_kv_lod = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cum_kv_lod[1:] = torch.cumsum(forward_batch.seq_lens, dim=0)
            self.cum_kv_lod_cpu = torch.zeros(
                (bs + 1,), dtype=torch.int32, device="cpu"
            )
            self.cum_kv_lod_cpu[1:] = torch.cumsum(forward_batch.seq_lens_cpu, dim=0)
            self.max_seq_q = max(forward_batch.extend_seq_lens_cpu)
            self.max_seq_kv = forward_batch.seq_lens_cpu.max().item()

            max_seqlen_pad = (
                forward_batch.seq_lens_cpu.max().item() + self.page_size - 1
            ) // self.page_size
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad), -1, dtype=torch.int32, device=self.device
            )
            create_kv_indices_native(
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_kv_indices,
                max_seqlen_pad,
                self.page_size,
            )
            self.forward_metadata = KunlunxinForwardMetadata(
                block_kv_indices=block_kv_indices
            )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        if k is None and v is None:
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            page_ids = forward_batch.out_cache_loc // self.page_size
            offsets = forward_batch.out_cache_loc % self.page_size
            k_paged = _as_paged_cache(
                k_buf, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_paged = _as_paged_cache(
                v_buf, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            k = k_paged[page_ids, offsets, :, :]
            v = v_paged[page_ids, offsets, :, :]

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = not (
            layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
        )
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        _sdpa_forward_extend_torch(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        if sinks is not None:
            sinks = sinks.float().contiguous()

        if k is None and v is None:
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            page_ids = forward_batch.out_cache_loc // self.page_size
            offsets = forward_batch.out_cache_loc % self.page_size
            k_paged = _as_paged_cache(
                k_buf, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_paged = _as_paged_cache(
                v_buf, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            k = k_paged[page_ids, offsets, :, :]
            v = v_paged[page_ids, offsets, :, :]

        assert self.forward_metadata is not None
        assert self.forward_metadata.block_kv_indices is not None
        assert self.forward_metadata.context_len_tensor is not None

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = torch.empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        else:
            o = torch.empty_like(q)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        bs = min(
            forward_batch.batch_size,
            self.forward_metadata.context_len_tensor.shape[0],
            q_.shape[0],
        )
        block_table = self.forward_metadata.block_kv_indices[:bs]
        context_lens = self.forward_metadata.context_len_tensor[:bs]

        _paged_attention_decode_torch(
            query=q_[:bs],
            output=o_[:bs],
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            block_table=block_table,
            context_lens=context_lens,
            block_size=self.page_size,
            scale=layer.scaling,
            num_kv_heads=layer.tp_k_head_num,
            sinks=sinks,
        )
        if bs < o_.shape[0]:
            o_[bs:].zero_()
        return o

    def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
        # Cap Graph block-table width so one-shot PyTorch decode stays tractable
        # during capture. Eager decode still uses exact-sized tables.
        # 512 tokens is enough for short offline prompts while keeping capture fast
        # on Kunlun (FlagGems + large gathers are extremely slow to capture).
        _MAX_GRAPH_TOKENS = 512
        full_blocks = self.req_to_token.shape[1] // self.page_size
        max_num_blocks = min(
            full_blocks,
            max(1, (_MAX_GRAPH_TOKENS + self.page_size - 1) // self.page_size),
        )
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, max_num_blocks), -1, dtype=torch.int32, device=self.device
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf
        self.context_len_tensor_cuda_graph = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        # Keep full storage width so the Python page-loop trip count is fixed
        # for CUDA Graph capture / replay.
        max_seqlen_pad = self.cuda_graph_kv_indices.stride(0)
        self.cuda_graph_kv_indices[:bs].fill_(-1)
        create_kv_indices_native(
            self.req_to_token,
            req_pool_indices,
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            max_seqlen_pad,
            self.page_size,
        )
        self.context_len_tensor_cuda_graph[:bs].copy_(seq_lens.int())
        self.forward_metadata = KunlunxinForwardMetadata(
            block_kv_indices=self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            context_len_tensor=self.context_len_tensor_cuda_graph[:bs],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        out_cache_loc=None,
    ):
        seq_lens = seq_lens[:bs]
        # Same full width as capture — context_lens mask drops unused pages.
        max_seqlen_pad = self.cuda_graph_kv_indices.stride(0)
        self.cuda_graph_kv_indices[:bs].fill_(-1)
        create_kv_indices_native(
            self.req_to_token,
            req_pool_indices[:bs],
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            max_seqlen_pad,
            self.page_size,
        )
        self.context_len_tensor_cuda_graph[:bs].copy_(seq_lens.int())
        self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
            :bs, :max_seqlen_pad
        ]
        self.forward_metadata.context_len_tensor = self.context_len_tensor_cuda_graph[
            :bs
        ]

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
