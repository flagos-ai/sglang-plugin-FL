"""Kunlunxin OOT attention backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

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


class KunlunxinBackend(AttentionBackend):
    """Kunlunxin (klx) full-attention backend for GDN-hybrid full layers."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.forward_metadata: Optional[KunlunxinForwardMetadata] = None

        # Cumulative LODs, filled per prefill/extend forward.
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
            # Prefill / Extend
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
        import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.* on klx)

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        if sinks is not None:
            sinks = sinks.float().contiguous()

        # KV shared layer: k/v are None -> read from paged KV cache.
        if k is None and v is None:
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            page_ids = forward_batch.out_cache_loc // self.page_size
            offsets = forward_batch.out_cache_loc % self.page_size
            k = k_buf[page_ids, :, offsets, :]
            v = v_buf[page_ids, :, offsets, :]

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

        is_causal_mask = not (
            layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
        )
        if q_.shape[0] != self.cum_q_lod_cpu[-1]:
            o_.fill_(0)

        use_kv_cache = (
            forward_batch.extend_prefix_lens_cpu is not None
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        )

        if use_kv_cache:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).contiguous()
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).contiguous()
            k_cache_max, v_cache_max = None, None
            block_tables = self.forward_metadata.block_kv_indices
            eff_max_seq_kv = (
                self.max_seq_kv
                if self.max_seq_kv is not None
                else block_tables.shape[1] * self.page_size
            )
            max_blocks_per_seq = (eff_max_seq_kv + self.page_size - 1) // self.page_size
            prefill_len = self.max_seq_q
        else:
            k_cache = v_cache = k_cache_max = v_cache_max = block_tables = None
            max_blocks_per_seq = -1
            prefill_len = -1

        torch.ops.sgl_kernel.klx_attention_extend.default(
            out_tensor=o_,
            q_tensor=q_.contiguous(),
            k_tensor=k.contiguous(),
            v_tensor=v.contiguous(),
            k_cache=k_cache,
            v_cache=v_cache,
            k_cache_maxptr=k_cache_max,
            v_cache_maxptr=v_cache_max,
            block_tables=block_tables,
            batch_num=forward_batch.batch_size,
            max_seq_q=self.max_seq_q,
            max_seq_kv=self.max_seq_kv,
            head_num=layer.tp_q_head_num,
            head_dim=layer.head_dim,
            kv_head_num=layer.tp_k_head_num,
            cum_q_lod=self.cum_q_lod,
            cum_q_lod_cpu=self.cum_q_lod_cpu,
            cum_kv_lod=self.cum_kv_lod,
            cum_kv_lod_cpu=self.cum_kv_lod_cpu,
            is_causal_mask=is_causal_mask,
            vo_head_dim=layer.v_head_dim,
            block_size=self.page_size,
            alpha=layer.scaling * math.sqrt(layer.head_dim),
            max_blocks_per_seq=max_blocks_per_seq,
            prefill_len=prefill_len,
            attn_sink_tensor=sinks,
            swa_left_size=-1,
            swa_right_size=-1,
        )
        return o

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        import sgl_kernel  # noqa: F401

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        if sinks is not None:
            sinks = sinks.float().contiguous()

        if k is None and v is None:
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            page_ids = forward_batch.out_cache_loc // self.page_size
            offsets = forward_batch.out_cache_loc % self.page_size
            k = k_buf[page_ids, :, offsets, :]
            v = v_buf[page_ids, :, offsets, :]

        assert self.forward_metadata is not None

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
            layer.layer_id
        ).contiguous()
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
            layer.layer_id
        ).contiguous()

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
        o_4d = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        k_cache_max, v_cache_max = None, None
        bs = min(
            forward_batch.batch_size,
            self.forward_metadata.context_len_tensor.shape[0],
        )
        block_table = self.forward_metadata.block_kv_indices[:bs]
        max_num_blocks_per_seq = self.forward_metadata.block_kv_indices.stride(0)

        torch.ops.sgl_kernel.klx_attention_decode.default(
            out_tensor=o_4d,
            q_tensor=q_.contiguous(),
            k_cache_tensor=k_cache,
            v_cache_tensor=v_cache,
            num_kv_heads=layer.tp_k_head_num,
            scale=layer.scaling,
            block_table_tensor=block_table,
            context_len_tensor=self.forward_metadata.context_len_tensor[:bs],
            block_size=self.page_size,
            max_context_len=self.max_context_len,
            batch_num=bs,
            head_num=layer.tp_q_head_num,
            head_dim=layer.head_dim,
            max_num_blocks_per_seq=max_num_blocks_per_seq,
            v_head_dim=layer.v_head_dim,
            k_cache_max_tensor=k_cache_max,
            v_cache_max_tensor=v_cache_max,
            attn_sink_tensor=sinks,
            max_window_size=-1,
        )
        return o

    def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
        max_num_blocks = self.req_to_token.shape[1] // self.page_size
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, max_num_blocks), 1, dtype=torch.int32, device=self.device
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
        max_seqlen_pad = self.cuda_graph_kv_indices.stride(0)
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
        seq_lens_cpu = seq_lens_cpu[:bs]
        max_seqlen_pad = (
            seq_lens_cpu.max().item() + self.page_size - 1
        ) // self.page_size
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
