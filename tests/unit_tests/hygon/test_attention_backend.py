# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the strict HCU attention backend on SGLang 0.5.18.

The stubs implement the upstream graph API and page-block launch contract;
real tensor metadata and vendor call layouts are checked without GPU packages.
"""

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class Mode(Enum):
    DECODE = 1
    EXTEND = 2
    IDLE = 3

    def is_decode_or_idle(self):
        return self in (Mode.DECODE, Mode.IDLE)


class PageTableKernel:
    """Model the new kernel: one CTA per 4096-token block on grid axis 1."""

    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def launch(token_map, requests, lengths, start, target, *strides, PAGED_SIZE):
            assert start is None
            blocks = grid[1] if len(grid) > 1 else 1
            pages_per_block = 4096 // PAGED_SIZE
            for row in range(grid[0]):
                pages = (int(lengths[row]) + PAGED_SIZE - 1) // PAGED_SIZE
                written_pages = min(pages, blocks * pages_per_block)
                slots = torch.arange(written_pages) * PAGED_SIZE
                target[row, :written_pages] = (
                    token_map[int(requests[row]), slots] // PAGED_SIZE
                ).to(torch.int32)

        return launch


@pytest.fixture
def backend_module(monkeypatch):
    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    class TritonBackend:
        needs_cpu_seq_lens = False

        def __init__(self, runner, skip_prefill=False, kv_indptr_buf=None):
            self.use_mla = False
            self.dcp_size = getattr(runner, "dcp_size", 1)
            self._translate_kv_loc = getattr(runner, "translate_kv_loc", None)
            self.sliding_window_size = None
            self.req_to_token = runner.req_to_token_pool.req_to_token
            self.max_context_len = runner.model_config.context_len
            self.device = runner.device
            self.num_head = 2
            self.enable_deterministic = False
            self.eager_calls = []
            self.graph_calls = []

        def init_forward_metadata(self, forward_batch):
            self.eager_calls.append(forward_batch)

        def init_cuda_graph_state(self, *args, **kwargs):
            self.graph_state_args = args, kwargs

        def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
            self.graph_calls.append((forward_batch.batch_size, in_capture))

        def init_forward_metadata_in_graph(self, forward_batch):
            pass

        def get_cuda_graph_seq_len_fill_value(self):
            return 1

    kernel = PageTableKernel()
    module(
        "sglang.kernels.ops.kvcache.kv_indices",
        create_flashmla_kv_indices_triton=kernel,
        get_num_kv_index_blocks_flashmla=lambda width, page: (
            (width + 4096 // page - 1) // (4096 // page)
        ),
    )
    module(
        "sglang.srt.layers.attention.triton_backend", TritonAttnBackend=TritonBackend
    )
    module(
        "sglang.srt.layers.radix_attention",
        AttentionType=SimpleNamespace(DECODER="decoder"),
    )
    module("sglang.srt.model_executor.forward_batch_info", ForwardMode=Mode)
    module(
        "sglang.srt.model_executor.cuda_graph_config",
        Backend=SimpleNamespace(FULL="full", DISABLED="disabled"),
        Phase=SimpleNamespace(PREFILL="prefill", DECODE="decode"),
    )
    module(
        "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager",
        is_in_tc_piecewise_cuda_graph=lambda: False,
    )
    module(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context",
        is_in_breakable_cuda_graph=lambda: False,
    )
    module(
        "flash_attn",
        flash_attn_with_kvcache=lambda **kw: torch.zeros_like(kw["q"]),
        flash_attn_varlen_func=lambda **kw: torch.zeros_like(kw["q"]),
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "sglang_fl/dispatch/backends/vendor/hygon/impl/attention_backend.py"
    )
    spec = importlib.util.spec_from_file_location("hcu_attention_contract", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.test_page_kernel = kernel
    loaded.test_triton_backend = TritonBackend
    return loaded


def make_runner(page_size=64, context_len=8192, decode="full", prefill="disabled"):
    # Each request owns a distinct contiguous range; token zero is a sink.
    token_map = torch.stack(
        [
            torch.arange(page_size, page_size + context_len),
            torch.arange(page_size + context_len, page_size + 2 * context_len),
        ]
    )
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=token_map),
        device="cpu",
        kv_cache_dtype=torch.bfloat16,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, head_dim=64, v_head_dim=64, context_len=context_len
        ),
        server_args=SimpleNamespace(
            page_size=page_size,
            disable_radix_cache=True,
            enable_deterministic_inference=False,
            cuda_graph_config={
                "decode": SimpleNamespace(backend=decode),
                "prefill": SimpleNamespace(backend=prefill),
            },
        ),
    )


def make_batch(lengths, requests=None, dtype=torch.int64):
    return SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=Mode.DECODE,
        spec_info=None,
        seq_lens=torch.tensor(lengths, dtype=dtype),
        seq_lens_cpu=torch.tensor(lengths),
        req_pool_indices=torch.tensor(requests or list(range(len(lengths)))),
        extend_prefix_lens_cpu=None,
    )


@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_long_eager_sequences_fill_every_page_and_clear_stale_entries(
    backend_module, page_size
):
    backend = backend_module.HCUAttnBackend(make_runner(page_size))
    long_batch = make_batch([4097, 8192])
    backend.init_forward_metadata(long_batch)
    table = backend._flash_attn_forward_page_table
    assert backend_module.test_page_kernel.grids[-1] == (2, 2)
    diagnostics = backend_module._validate_flash_attn_page_table(
        backend.req_to_token,
        long_batch.req_pool_indices,
        long_batch.seq_lens,
        table,
        page_size=page_size,
        require_disjoint_pages=True,
    )
    assert len(diagnostics[1]["page_ids"]) == 8192 // page_size
    assert backend._flash_attn_forward_seq_lens.dtype == torch.int32

    # Reusing the largest buffer must clear the old tail for shorter requests.
    short_batch = make_batch([17, 33])
    backend.init_forward_metadata(short_batch)
    assert backend._flash_attn_forward_page_table.data_ptr() == table.data_ptr()
    assert torch.all(table[:, 1:] == -1)
    assert backend.eager_calls == [long_batch, short_batch]


def test_graph_capture_and_replay_refresh_stable_page_table(backend_module):
    backend = backend_module.HCUAttnBackend(make_runner())
    backend.init_cuda_graph_state(2, 2)
    batch = make_batch([4097, 129], requests=[1, 0], dtype=torch.int32)
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    table = backend._flash_attn_forward_page_table
    address = table.data_ptr()
    sequence_address = backend._flash_attn_forward_seq_lens.data_ptr()
    assert table[0, 64] == (8192 + 64) // 64 + 64

    batch.req_pool_indices.copy_(torch.tensor([0, 1]))
    batch.seq_lens.copy_(torch.tensor([65, 1]))
    backend.init_forward_metadata_out_graph(batch)
    assert backend._flash_attn_forward_page_table.data_ptr() == address
    assert backend._flash_attn_forward_seq_lens.data_ptr() == sequence_address
    assert table[0, :2].tolist() == [1, 2]
    assert table[1, 0] == (8192 + 64) // 64
    assert torch.all(table[0, 2:] == -1)
    assert backend.graph_calls == [(2, True), (2, False)]
    assert backend.get_cuda_graph_seq_len_fill_value() == 1
    backend.init_forward_metadata_in_graph(batch)

    # Legacy hooks were removed upstream and must not be the integration point.
    assert "init_forward_metadata_capture_cuda_graph" not in type(backend).__dict__
    assert "init_forward_metadata_replay_cuda_graph" not in type(backend).__dict__


@pytest.mark.parametrize("decode", ["full", "disabled"])
def test_new_graph_config_and_cpu_mirror_contract(backend_module, decode):
    backend = backend_module.HCUAttnBackend(make_runner(decode=decode))
    assert isinstance(backend, backend_module.test_triton_backend)
    assert backend.needs_cpu_seq_lens is True
    assert not hasattr(make_runner().server_args, "disable_piecewise_cuda_graph")


@pytest.mark.parametrize(
    "decode,prefill,message",
    [
        ("full", "full", "prefill.backend must be disabled"),
        ("full", "tc_piecewise", "prefill.backend must be disabled"),
        ("full", "breakable", "prefill.backend must be disabled"),
        ("tc_piecewise", "disabled", "decode.backend must be full or disabled"),
        ("breakable", "disabled", "decode.backend must be full or disabled"),
    ],
)
def test_rejects_unsupported_graph_modes(backend_module, decode, prefill, message):
    with pytest.raises(RuntimeError, match=message):
        backend_module.HCUAttnBackend(make_runner(decode=decode, prefill=prefill))


@pytest.mark.parametrize("feature", ["dcp", "virtual_kv"])
def test_rejects_cache_features_not_supported_by_vendor_kernel(backend_module, feature):
    runner = make_runner()
    if feature == "dcp":
        runner.dcp_size = 2
        message = "decode context parallelism"
    else:
        runner.translate_kv_loc = lambda indices: indices
        message = "virtual cache locations"
    with pytest.raises(RuntimeError, match=message):
        backend_module.HCUAttnBackend(runner)


def test_new_forward_options_fail_before_vendor_kernel(backend_module):
    backend = backend_module.HCUAttnBackend(make_runner())
    for method in (backend.forward_decode, backend.forward_extend):
        for kwargs in ({"score_mod": lambda *args: None}, {"aux_tensors": ()}):
            with pytest.raises(RuntimeError, match="score_mod or aux_tensors"):
                method(None, None, None, None, None, **kwargs)


def test_prefill_metadata_uses_cpu_mirror_and_int32_cumulative_lengths(backend_module):
    backend = backend_module.HCUAttnBackend(make_runner())
    batch = make_batch([2, 3])
    batch.forward_mode = Mode.EXTEND
    batch.extend_prefix_lens_cpu = [0, 0]
    batch.extend_seq_lens_cpu = [2, 3]
    batch.extend_seq_lens = torch.tensor([2, 3])
    backend.init_forward_metadata(batch)
    assert backend._flash_attn_prefill_metadata_ready
    assert backend._flash_attn_prefill_cu_seqlens_q.tolist() == [0, 2, 5]
    assert backend._flash_attn_prefill_cu_seqlens_q.dtype == torch.int32
    assert backend._flash_attn_prefill_max_k_seq_len == 3
    assert not backend._flash_attn_prefill_uses_paged_kv


def test_graph_rejects_extend_and_speculative_modes(backend_module):
    backend = backend_module.HCUAttnBackend(make_runner())
    backend.init_cuda_graph_state(2, 2)
    batch = make_batch([2, 3], dtype=torch.int32)
    batch.forward_mode = Mode.EXTEND
    with pytest.raises(RuntimeError, match="only normal decode"):
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
    batch.forward_mode = Mode.DECODE
    batch.spec_info = object()
    with pytest.raises(RuntimeError, match="only normal decode"):
        backend.init_forward_metadata_out_graph(batch)


class KVPool:
    def __init__(self):
        self.k = torch.zeros((16448, 1, 64), dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.writes = 0

    def get_kv_buffer(self, layer_id):
        assert layer_id == 0
        return self.k, self.v

    def set_kv_buffer(self, layer, locations, k, v):
        self.k[locations] = k
        self.v[locations] = v
        self.writes += 1


def make_layer():
    return SimpleNamespace(
        layer_id=0,
        qk_head_dim=64,
        v_head_dim=64,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        k_scale=None,
        v_scale=None,
        scaling=0.125,
        logit_cap=0.0,
    )


@pytest.mark.parametrize("has_prefix", [False, True])
def test_prefill_keeps_vendor_varlen_and_paged_cache_layout(backend_module, has_prefix):
    backend = backend_module.HCUAttnBackend(make_runner())
    prefix = [4096, 64] if has_prefix else [0, 0]
    extend = [1, 1] if has_prefix else [2, 3]
    full = [p + e for p, e in zip(prefix, extend)]
    batch = make_batch(full)
    batch.forward_mode = Mode.EXTEND
    batch.extend_prefix_lens_cpu = prefix
    batch.extend_seq_lens_cpu = extend
    batch.extend_seq_lens = torch.tensor(extend)
    batch.token_to_kv_pool = KVPool()
    batch.out_cache_loc = torch.cat(
        [
            backend.req_to_token[i, p : p + e]
            for i, (p, e) in enumerate(zip(prefix, extend))
        ]
    )
    backend.init_forward_metadata(batch)
    num_tokens = sum(extend)
    q = torch.ones((num_tokens, 128), dtype=torch.bfloat16)
    k = torch.ones((num_tokens, 1, 64), dtype=torch.bfloat16)
    v = k * 2
    calls = []

    def vendor_varlen(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"].shape == (num_tokens, 2, 64)
        assert kwargs["cu_seqlens_q"].dtype == torch.int32
        assert kwargs["cu_seqlens_k"].dtype == torch.int32
        assert kwargs["causal"] is True
        assert kwargs["max_seqlen_k"] == max(full)
        if has_prefix:
            assert kwargs["k"].shape == (257, 64, 1, 64)
            assert kwargs["block_table"][0, 64] == 65
            torch.testing.assert_close(kwargs["k"][65, 0], k[0])
            torch.testing.assert_close(kwargs["v"][65, 0], v[0])
        else:
            assert "block_table" not in kwargs
            torch.testing.assert_close(kwargs["k"], k)
            torch.testing.assert_close(kwargs["v"], v)
        return torch.full_like(kwargs["q"], 7)

    backend._flash_attn_varlen_func = vendor_varlen
    output = backend.forward_extend(q, k, v, make_layer(), batch, score_mod=None)
    assert len(calls) == 1
    assert batch.token_to_kv_pool.writes == 1
    assert output.shape == (num_tokens, 128)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 7)


def test_decode_keeps_vendor_paged_cache_layout(backend_module):
    backend = backend_module.HCUAttnBackend(make_runner())
    batch = make_batch([4097, 65])
    batch.token_to_kv_pool = KVPool()
    batch.out_cache_loc = torch.stack(
        [backend.req_to_token[i, length - 1] for i, length in enumerate([4097, 65])]
    )
    backend.init_forward_metadata(batch)
    q = torch.ones((2, 128), dtype=torch.bfloat16)
    k = torch.ones((2, 1, 64), dtype=torch.bfloat16)
    v = k * 2
    calls = []

    def vendor_decode(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"].shape == (2, 1, 2, 64)
        assert kwargs["k_cache"].shape == (257, 64, 1, 64)
        assert kwargs["v_cache"].shape == (257, 64, 1, 64)
        assert kwargs["cache_seqlens"].dtype == torch.int32
        assert kwargs["block_table"][0, 64] == 65
        assert kwargs["num_splits"] == 1
        assert kwargs["causal"] is True
        torch.testing.assert_close(kwargs["k_cache"][65, 0], k[0])
        torch.testing.assert_close(kwargs["v_cache"][65, 0], v[0])
        return torch.full_like(kwargs["q"], 9)

    backend._flash_attn_with_kvcache = vendor_decode
    output = backend.forward_decode(q, k, v, make_layer(), batch, aux_tensors=None)
    assert len(calls) == 1
    assert batch.token_to_kv_pool.writes == 1
    assert output.shape == (2, 128)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 9)
