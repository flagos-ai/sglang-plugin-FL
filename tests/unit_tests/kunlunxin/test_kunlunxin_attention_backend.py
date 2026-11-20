# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ``impl/attention_backend.py`` (klx full attention).

This is the OOT attention backend that serves the full-attention layers of the
Qwen3.5/3.6 GDN hybrid. The klx kernels are stubbed, so the tests pin the
metadata preparation and the exact launch contract SGLang 0.5.18 relies on:

  * ``req_to_token`` -> page-block table conversion,
  * prefill/extend LODs and the decode ``context_len`` tensor,
  * ``alpha = scaling * sqrt(head_dim)`` and causal-mask derivation,
  * the shared-KV path (k/v read from the paged cache when ``k is None``),
  * the 0.5.18 out-graph capture/replay hook that replaced the deprecated
    split capture/replay entry points.
"""

import math
import sys
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import AttentionType

from sglang_fl.dispatch.backends.vendor.kunlunxin.impl import attention_backend as attn

PAGE_SIZE = 2
NUM_REQS = 4
TOKENS_PER_REQ = 16


class _KernelNamespace:
    def __init__(self):
        self.calls = []

    def default(self, **kwargs):
        self.calls.append(kwargs)


class _KVPool:
    def __init__(self, pages=8, kv_heads=1, page_size=PAGE_SIZE, head_dim=4):
        torch.manual_seed(0)
        self.k = torch.randn(pages, kv_heads, page_size, head_dim)
        self.v = torch.randn(pages, kv_heads, page_size, head_dim)
        self.sets = []

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v

    def set_kv_buffer(self, layer, out_cache_loc, k, v, **kwargs):
        self.sets.append((out_cache_loc, k, v))


def _make_runner(context_len=512):
    runner = SimpleNamespace()
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(context_len=context_len)
    runner.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(NUM_REQS * TOKENS_PER_REQ).reshape(
            NUM_REQS, TOKENS_PER_REQ
        )
    )
    runner.token_to_kv_pool = _KVPool()
    runner.kv_cache_dtype = torch.float32
    runner.page_size = PAGE_SIZE
    return runner


def _install_kernels(monkeypatch):
    extend = _KernelNamespace()
    decode = _KernelNamespace()
    monkeypatch.setattr(
        torch.ops,
        "sgl_kernel",
        SimpleNamespace(klx_attention_extend=extend, klx_attention_decode=decode),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "sgl_kernel", SimpleNamespace())
    return extend, decode


def _decode_batch(seq_lens=(5, 9), reqs=(0, 1)):
    return SimpleNamespace(
        batch_size=len(seq_lens),
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        req_pool_indices=torch.tensor(reqs),
    )


def _prefill_batch(extend=(3, 5), total=(3, 8), reqs=(1, 2)):
    return SimpleNamespace(
        batch_size=len(extend),
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False),
        extend_seq_lens=torch.tensor(extend),
        extend_seq_lens_cpu=list(extend),
        seq_lens=torch.tensor(total),
        seq_lens_cpu=torch.tensor(total),
        req_pool_indices=torch.tensor(reqs),
    )


def _layer(**overrides):
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
        head_dim=4,
        scaling=0.5,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
    )
    layer.__dict__.update(overrides)
    return layer


# ─── metadata ────────────────────────────────────────────────────────────────


def test_backend_exposes_the_pools_upstream_reads_unconditionally():
    runner = _make_runner()
    backend = attn.KunlunxinBackend(runner)

    assert backend.device == torch.device("cpu")
    assert backend.max_context_len == 512
    assert backend.page_size == PAGE_SIZE
    assert backend.req_to_token is runner.req_to_token_pool.req_to_token
    assert backend.req_to_token_pool is runner.req_to_token_pool
    assert backend.token_to_kv_pool is runner.token_to_kv_pool


def test_create_kv_indices_native_converts_tokens_to_page_blocks():
    req_to_token = torch.arange(NUM_REQS * TOKENS_PER_REQ).reshape(
        NUM_REQS, TOKENS_PER_REQ
    )
    target = torch.full((2, 4), -1, dtype=torch.int32)

    attn.create_kv_indices_native(
        req_to_token, torch.tensor([1]), target, max_seqlen_pad=4, page_size=PAGE_SIZE
    )

    # Row 1 owns tokens 16..31 -> blocks 8,9,10,11 at the page starts.
    expected = (req_to_token[1, : 4 * PAGE_SIZE].reshape(4, PAGE_SIZE)[:, 0] // PAGE_SIZE)
    assert torch.equal(target[0], expected.int())


def test_decode_metadata_builds_page_table_and_context_lengths():
    backend = attn.KunlunxinBackend(_make_runner())

    backend.init_forward_metadata(_decode_batch())

    metadata = backend.forward_metadata
    # max_seqlen_pad = ceil(9 / 2) = 5
    assert metadata.block_kv_indices.shape == (2, 5)
    assert metadata.block_kv_indices.dtype is torch.int32
    assert metadata.context_len_tensor.tolist() == [5, 9]
    assert metadata.context_len_tensor.dtype is torch.int32
    assert metadata.context_len_tensor.is_contiguous()
    expected_blocks = (
        backend.req_to_token[1, : 5 * PAGE_SIZE].reshape(5, PAGE_SIZE)[:, 0] // PAGE_SIZE
    )
    assert torch.equal(metadata.block_kv_indices[1], expected_blocks.int())


def test_prefill_metadata_builds_lods():
    backend = attn.KunlunxinBackend(_make_runner())

    backend.init_forward_metadata(_prefill_batch())

    assert backend.cum_q_lod.tolist() == [0, 3, 8]
    assert backend.cum_q_lod_cpu.tolist() == [0, 3, 8]
    assert backend.cum_kv_lod.tolist() == [0, 3, 11]
    assert backend.cum_kv_lod_cpu.tolist() == [0, 3, 11]
    assert backend.max_seq_q == 5
    assert backend.max_seq_kv == 8
    assert backend.forward_metadata.block_kv_indices.shape == (2, 4)
    assert backend.cum_q_lod.dtype is torch.int32
    assert backend.cum_q_lod_cpu.device.type == "cpu"


# ─── forward ─────────────────────────────────────────────────────────────────


def _extend_batch(prefix_lens, extend=(3, 5), total=(3, 8)):
    batch = _prefill_batch(extend=extend, total=total)
    batch.extend_prefix_lens_cpu = prefix_lens
    # Non-contiguous on purpose: the backend must normalise out_cache_loc.
    batch.out_cache_loc = torch.arange(16)[::2]
    return batch


def test_forward_extend_launches_klx_with_the_declared_arguments(monkeypatch):
    extend, _ = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _extend_batch([0, 0])
    batch.batch_size = 2
    backend.init_forward_metadata(batch)
    tokens = int(backend.cum_q_lod_cpu[-1])
    q = torch.randn(tokens, 2 * 4)
    k = torch.randn(tokens, 4)
    v = torch.randn(tokens, 4)
    layer = _layer()

    out = backend.forward_extend(q, k, v, layer, batch)

    call = extend.calls[-1]
    assert out.shape == (tokens, 2 * 4)
    assert call["batch_num"] == 2
    assert call["max_seq_q"] == 5
    assert call["max_seq_kv"] == 8
    assert call["head_num"] == 2
    assert call["kv_head_num"] == 1
    assert call["vo_head_dim"] == 4
    assert call["block_size"] == PAGE_SIZE
    assert call["alpha"] == layer.scaling * math.sqrt(layer.head_dim)
    assert call["is_causal_mask"] is True
    # No prefix tokens cached yet -> the kernel must not be handed a page table.
    assert call["block_tables"] is None
    assert call["max_blocks_per_seq"] == -1
    assert call["prefill_len"] == -1
    assert call["swa_left_size"] == -1 and call["swa_right_size"] == -1
    assert call["attn_sink_tensor"] is None
    assert batch.out_cache_loc.is_contiguous()


def test_forward_extend_uses_the_page_table_when_a_prefix_is_cached(monkeypatch):
    extend, _ = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _extend_batch([2, 3])
    batch.batch_size = 2
    backend.init_forward_metadata(batch)
    tokens = int(backend.cum_q_lod_cpu[-1])

    backend.forward_extend(
        torch.randn(tokens, 8), torch.randn(tokens, 4), torch.randn(tokens, 4),
        _layer(), batch,
    )

    call = extend.calls[-1]
    assert call["block_tables"] is backend.forward_metadata.block_kv_indices
    assert call["max_blocks_per_seq"] == (backend.max_seq_kv + PAGE_SIZE - 1) // PAGE_SIZE
    assert call["prefill_len"] == backend.max_seq_q
    assert call["k_cache"] is not None and call["v_cache"] is not None


def test_forward_extend_reads_kv_from_the_cache_for_shared_layers(monkeypatch):
    extend, _ = _install_kernels(monkeypatch)
    runner = _make_runner()
    backend = attn.KunlunxinBackend(runner)
    batch = _extend_batch([0, 0])
    batch.batch_size = 2
    backend.init_forward_metadata(batch)
    tokens = int(backend.cum_q_lod_cpu[-1])

    backend.forward_extend(torch.randn(tokens, 8), None, None, _layer(), batch)

    # Shared (cross-layer) attention: k/v are gathered from the paged cache and
    # written back through set_kv_buffer.
    loc, k_written, v_written = runner.token_to_kv_pool.sets[-1]
    assert k_written.shape == (loc.numel(), 1, 4)
    assert v_written.shape == (loc.numel(), 1, 4)
    assert extend.calls[-1]["k_tensor"].shape[0] == loc.numel()


def test_forward_extend_casts_sinks_and_clears_the_causal_mask_for_cross_attention(
    monkeypatch,
):
    extend, _ = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _extend_batch([0, 0])
    batch.batch_size = 2
    backend.init_forward_metadata(batch)
    tokens = int(backend.cum_q_lod_cpu[-1])
    sinks = torch.randn(2, dtype=torch.float64)

    backend.forward_extend(
        torch.randn(tokens, 8),
        torch.randn(tokens, 4),
        torch.randn(tokens, 4),
        _layer(is_cross_attention=True),
        batch,
        sinks=sinks,
    )

    call = extend.calls[-1]
    assert call["is_causal_mask"] is False
    assert call["attn_sink_tensor"].dtype is torch.float32
    assert call["attn_sink_tensor"].is_contiguous()


def test_forward_decode_launches_klx_with_metadata_slices(monkeypatch):
    _, decode = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _decode_batch()
    batch.out_cache_loc = torch.arange(8)
    backend.init_forward_metadata(batch)
    layer = _layer()

    out = backend.forward_decode(
        torch.randn(2, 2 * 4), torch.randn(2, 4), torch.randn(2, 4), layer, batch
    )

    call = decode.calls[-1]
    metadata = backend.forward_metadata
    assert out.shape == (2, 2 * 4)
    assert call["batch_num"] == 2
    assert call["scale"] == layer.scaling
    assert call["block_size"] == PAGE_SIZE
    assert call["max_context_len"] == 512
    assert call["max_num_blocks_per_seq"] == metadata.block_kv_indices.stride(0)
    assert torch.equal(call["context_len_tensor"], metadata.context_len_tensor[:2])
    assert call["head_num"] == 2
    assert call["num_kv_heads"] == 1
    assert call["head_dim"] == 4
    assert call["v_head_dim"] == 4
    assert call["max_window_size"] == -1


def test_forward_decode_honours_a_diverging_v_head_dim(monkeypatch):
    _, decode = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _decode_batch()
    batch.out_cache_loc = torch.arange(8)
    backend.init_forward_metadata(batch)

    out = backend.forward_decode(
        torch.randn(2, 2 * 4),
        torch.randn(2, 2),
        torch.randn(2, 2),
        _layer(tp_k_head_num=2, v_head_dim=2),
        batch,
    )

    assert out.shape == (2, 2 * 2)
    assert decode.calls[-1]["v_head_dim"] == 2


def test_forward_decode_clamps_the_batch_to_the_metadata_buffer(monkeypatch):
    _, decode = _install_kernels(monkeypatch)
    backend = attn.KunlunxinBackend(_make_runner())
    batch = _decode_batch()
    batch.out_cache_loc = torch.arange(8)
    backend.init_forward_metadata(batch)
    # A stale/short context buffer must drive the launched batch size.
    backend.forward_metadata.context_len_tensor = (
        backend.forward_metadata.context_len_tensor[:1]
    )

    backend.forward_decode(
        torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 4), _layer(), batch
    )

    assert decode.calls[-1]["batch_num"] == 1
    assert decode.calls[-1]["block_table_tensor"].shape[0] == 1


# ─── cuda graph (out_graph API) ──────────────────────────────────────────────


def test_cuda_graph_state_allocation():
    backend = attn.KunlunxinBackend(_make_runner())

    backend.init_cuda_graph_state(max_bs=3, max_num_tokens=8)

    assert backend.cuda_graph_kv_indices.shape == (3, TOKENS_PER_REQ // PAGE_SIZE)
    assert backend.cuda_graph_kv_indices.dtype is torch.int32
    assert backend.context_len_tensor_cuda_graph.shape == (3,)
    assert backend.get_cuda_graph_seq_len_fill_value() == 1


def test_out_graph_capture_and_replay_refill_the_static_buffers():
    backend = attn.KunlunxinBackend(_make_runner())
    backend.init_cuda_graph_state(max_bs=2, max_num_tokens=8)
    stride = backend.cuda_graph_kv_indices.stride(0)

    backend.init_forward_metadata_out_graph(_decode_batch((5, 9)), in_capture=True)

    metadata = backend.forward_metadata
    assert metadata.block_kv_indices.shape == (2, stride)
    assert metadata.block_kv_indices.data_ptr() == backend.cuda_graph_kv_indices.data_ptr()
    assert metadata.context_len_tensor.tolist() == [5, 9]

    backend.init_forward_metadata_out_graph(_decode_batch((3, 4)), in_capture=False)

    metadata = backend.forward_metadata
    assert metadata.block_kv_indices.shape == (2, (4 + PAGE_SIZE - 1) // PAGE_SIZE)
    assert metadata.block_kv_indices.stride(0) == stride  # still a static-buffer view
    assert metadata.context_len_tensor.tolist() == [3, 4]
    expected = (
        backend.req_to_token[1, : 2 * PAGE_SIZE].reshape(2, PAGE_SIZE)[:, 0]
        // PAGE_SIZE
    )
    assert torch.equal(metadata.block_kv_indices[1], expected.int())
