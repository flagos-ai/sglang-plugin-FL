# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Speculative attention contracts for the top-k=1 Hygon MTP path.

Vendor calls are observed here; actual arithmetic is checked in the separate
GPU functional suite. These checks use the pinned SGLang 0.5.18 ownership API.
"""

from dataclasses import dataclass, replace
from enum import Enum
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.hygon import test_attention_backend as hcu_contracts

from tests.unit_tests.hygon.test_attention_backend import (
    make_layer,
    make_runner,
)


@pytest.fixture
def backend_module(monkeypatch, metadata_module):
    # Reuse pinned upstream stubs without initializing vendor registration.
    monkeypatch.setitem(
        sys.modules,
        "sglang_fl.dispatch.backends.vendor.hygon.impl.mtp_metadata",
        metadata_module,
    )
    return hcu_contracts.backend_module.__wrapped__(monkeypatch)


class SpecMode(Enum):
    DECODE = 1
    EXTEND = 2
    IDLE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND = 5
    DRAFT_EXTEND_V2 = 6

    def is_decode_or_idle(self):
        return self in (self.DECODE, self.IDLE)

    def is_target_verify(self):
        return self == self.TARGET_VERIFY

    def is_draft_extend(self, include_v2=False):
        return self == self.DRAFT_EXTEND or (
            include_v2 and self == self.DRAFT_EXTEND_V2
        )

    def is_draft_extend_v2(self):
        return self == self.DRAFT_EXTEND_V2

    def is_extend(self):
        return not self.is_decode_or_idle()

    def is_idle(self):
        return self == self.IDLE


@pytest.fixture
def metadata_module():
    path = Path(__file__).resolve().parents[3] / (
        "sglang_fl/dispatch/backends/vendor/hygon/impl/mtp_metadata.py"
    )
    spec = importlib.util.spec_from_file_location("hcu_mtp_metadata_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_mtp_backend(module, monkeypatch, *, topk=1):
    monkeypatch.setattr(module, "ForwardMode", SpecMode)
    runner = make_runner(decode="disabled")
    runner.server_args.speculative_eagle_topk = topk
    runner.server_args.speculative_num_steps = 3
    runner.server_args.speculative_num_draft_tokens = 4
    runner.server_args.speculative_algorithm = "EAGLE"
    runner.server_args.disable_overlap_schedule = True
    return module.HCUAttnBackend(runner)


def make_spec_batch(backend, prefix=(63, 65), width=4):
    prefix = list(prefix)
    requests = torch.arange(len(prefix))
    locations = torch.cat(
        [backend.req_to_token[i, n : n + width] for i, n in enumerate(prefix)]
    )
    return SimpleNamespace(
        batch_size=len(prefix),
        forward_mode=SpecMode.TARGET_VERIFY,
        global_forward_mode=None,
        spec_info=SimpleNamespace(draft_token_num=width, custom_mask=None),
        seq_lens=torch.tensor(prefix, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(prefix, dtype=torch.int64),
        seq_lens_sum=sum(prefix),
        req_pool_indices=requests,
        out_cache_loc=locations,
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=None,
        extend_seq_lens=None,
        encoder_lens=None,
    )


def test_spec_kv_indices_allow_nonconsecutive_physical_pages(metadata_module):
    # Contiguity is required inside a page, not between physical pages.
    indices = torch.cat(
        [torch.arange(448, 512), torch.arange(192, 194), torch.arange(320, 323)]
    )
    ptr = torch.tensor([0, 66, 69], dtype=torch.int32)
    table, lengths = metadata_module.kv_indices_to_page_table(
        ptr, indices, page_size=64, max_pages=3
    )
    assert table.dtype == lengths.dtype == torch.int32
    assert table.tolist() == [[7, 3, -1], [5, -1, -1]]
    assert lengths.tolist() == [66, 3]


def test_spec_kv_indices_refresh_output_in_place_and_clear_tail(metadata_module):
    out = torch.full((2, 4), 999, dtype=torch.int32)
    ptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    indices = torch.tensor([64, 192, 193])
    table, lengths = metadata_module.kv_indices_to_page_table(
        ptr, indices, page_size=64, max_pages=4, out=out
    )
    assert table.data_ptr() == out.data_ptr()
    assert table.tolist() == [[1, -1, -1, -1], [3, -1, -1, -1]]
    assert lengths.tolist() == [1, 2]


@pytest.mark.parametrize(
    "ptr,indices",
    [
        ([1, 3], [64, 65, 66]),
        ([0, 3, 2], [64, 65]),
        ([0, 4], [64, 65, 66]),
        ([0, 2], [65, 66]),
        ([0, 3], [64, 66, 67]),
        ([0, 2], [64, -1]),
    ],
)
def test_spec_kv_indices_reject_invalid_physical_mapping(metadata_module, ptr, indices):
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        metadata_module.kv_indices_to_page_table(
            torch.tensor(ptr, dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int64),
            page_size=64,
            max_pages=4,
        )


def test_spec_kv_indices_reject_insufficient_capacity(metadata_module):
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        metadata_module.kv_indices_to_page_table(
            torch.tensor([0, 65], dtype=torch.int32),
            torch.arange(64, 129),
            page_size=64,
            max_pages=1,
        )


def test_target_verify_counts_prefix_plus_all_draft_tokens(backend_module, monkeypatch):
    backend = make_mtp_backend(backend_module, monkeypatch)
    batch = make_spec_batch(backend)
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    assert backend._flash_attn_speculative_phase == "target_verify"
    assert backend._flash_attn_prefill_cu_seqlens_q.tolist() == [0, 4, 8]
    assert backend._flash_attn_prefill_cu_seqlens_k.tolist() == [0, 67, 136]
    assert backend._flash_attn_prefill_max_seq_len == 4
    assert backend._flash_attn_prefill_max_k_seq_len == 69
    # Metadata normalization may use a view; scheduler-owned lengths stay prefix-only.
    assert batch.seq_lens.tolist() == [63, 65]
    assert batch.seq_lens_cpu.tolist() == [63, 65]


def test_target_verify_passes_full_paged_chain_to_vendor(backend_module, monkeypatch):
    backend = make_mtp_backend(backend_module, monkeypatch)
    batch = make_spec_batch(backend)
    backend._prepare_flash_attn_speculative_metadata(batch)
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        assert kwargs["cu_seqlens_q"].tolist() == [0, 4, 8]
        assert kwargs["cu_seqlens_k"].tolist() == [0, 67, 136]
        assert kwargs["max_seqlen_q"] == 4
        assert kwargs["max_seqlen_k"] == 69
        assert kwargs["causal"] is True
        assert kwargs["block_table"].shape[0] == 2
        return torch.ones_like(kwargs["q"])

    backend._flash_attn_varlen_func = vendor
    q = torch.ones(8, 128, dtype=torch.bfloat16)
    k = torch.ones(8, 1, 64, dtype=torch.bfloat16)
    output = backend.forward_extend(q, k, k, make_layer(), batch)
    assert output.shape == q.shape
    assert len(calls) == 1
    assert backend.token_to_kv_pool.writes == 1
    torch.testing.assert_close(backend.token_to_kv_pool.k[batch.out_cache_loc], k)


def test_draft_decode_uses_per_step_spec_kv_indices(backend_module, monkeypatch):
    backend = make_mtp_backend(backend_module, monkeypatch)
    batch = make_spec_batch(backend)
    batch.forward_mode = SpecMode.DECODE
    # Later draft steps append to the target prefix without changing fb.seq_lens.
    kv_lens = [65, 68]
    batch.spec_info = SimpleNamespace(
        kv_indptr=torch.tensor([0, 65, 133], dtype=torch.int32),
        kv_indices=torch.cat(
            [backend.req_to_token[i, :n] for i, n in enumerate(kv_lens)]
        ),
    )
    batch.out_cache_loc = torch.stack(
        [backend.req_to_token[i, n - 1] for i, n in enumerate(kv_lens)]
    )
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    assert backend._flash_attn_speculative_phase == "draft_decode"
    assert backend._flash_attn_forward_seq_lens.tolist() == kv_lens
    assert backend._flash_attn_forward_seq_lens.dtype == torch.int32
    assert batch.seq_lens.tolist() == [63, 65]


def test_tree_topk_is_rejected_instead_of_causal_approximation(
    backend_module, monkeypatch
):
    with pytest.raises(RuntimeError, match="top.?k|topk"):
        make_mtp_backend(backend_module, monkeypatch, topk=2)


def test_speculative_graph_remains_explicitly_rejected(backend_module, monkeypatch):
    backend = make_mtp_backend(backend_module, monkeypatch)
    backend.init_cuda_graph_state(2, 8)
    batch = make_spec_batch(backend)
    with pytest.raises(RuntimeError, match="normal decode|speculative|MTP"):
        backend.init_forward_metadata_out_graph(batch, in_capture=True)


@pytest.mark.parametrize("accepted", [(1, 1), (2, 1), (4, 3)])
def test_draft_extend_v2_keeps_static_window_not_accepted_length(
    backend_module, monkeypatch, accepted
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    prefix, width = [70, 64], 4
    batch = make_spec_batch(backend, prefix, width)
    batch.forward_mode = SpecMode.DRAFT_EXTEND_V2
    batch.seq_lens = torch.tensor([74, 68], dtype=torch.int64)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.extend_prefix_lens_cpu = prefix
    batch.extend_prefix_lens = torch.tensor(prefix)
    batch.extend_seq_lens_cpu = [width, width]
    batch.extend_seq_lens = torch.tensor([width, width])
    batch.extend_start_loc = torch.tensor([0, width])
    batch.extend_num_tokens = 2 * width
    batch.spec_info = SimpleNamespace(
        num_tokens_per_req=width,
        extend_seq_lens_tensor=torch.tensor([width, width]),
        num_accept_tokens=torch.tensor(accepted),
    )
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    assert backend._flash_attn_speculative_phase == "draft_extend"
    assert backend._flash_attn_prefill_cu_seqlens_q.tolist() == [0, 4, 8]
    assert backend._flash_attn_prefill_cu_seqlens_k.tolist() == [0, 74, 142]
    assert backend._flash_attn_prefill_max_seq_len == 4
    assert batch.extend_seq_lens.tolist() == [4, 4]
    assert batch.spec_info.num_accept_tokens.tolist() == list(accepted)


def test_target_verify_noncausal_mask_is_not_silently_discarded(
    backend_module, monkeypatch
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    batch = make_spec_batch(backend)
    # The first query must not see the final draft token. Setting every edge
    # visible is a different attention problem, even with configured topk=1.
    batch.spec_info.custom_mask = torch.ones(4 * (67 + 69), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="mask|causal|chain|tree"):
        backend._prepare_flash_attn_speculative_metadata(batch)
    assert backend.token_to_kv_pool.writes == 0


@pytest.mark.parametrize("prefix", [(0, 0), (0, 64)])
def test_initial_draft_prefill_preserves_full_chunk_not_num_tokens_per_req(
    backend_module, monkeypatch, prefix
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    prefix, extend = list(prefix), [7, 5]
    batch = make_spec_batch(backend)
    batch.forward_mode = SpecMode.EXTEND
    batch.seq_lens = torch.tensor(
        [p + e for p, e in zip(prefix, extend)], dtype=torch.int64
    )
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.extend_prefix_lens_cpu = prefix
    batch.extend_prefix_lens = torch.tensor(prefix)
    batch.extend_seq_lens_cpu = extend
    batch.extend_seq_lens = torch.tensor(extend)
    batch.extend_start_loc = torch.tensor([0, 7])
    batch.extend_num_tokens = 12
    batch.out_cache_loc = torch.cat(
        [
            backend.req_to_token[i, p : p + e]
            for i, (p, e) in enumerate(zip(prefix, extend))
        ]
    )
    # EagleDraftExtendInput describes hidden-state consumption, not the actual
    # attention chunk width. The ordinary scheduler metadata is authoritative.
    batch.spec_info = SimpleNamespace(num_tokens_per_req=1)
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    assert backend._flash_attn_prefill_cu_seqlens_q.tolist() == [0, 7, 12]
    assert backend._flash_attn_prefill_cu_seqlens_k.tolist() == [
        0,
        7,
        sum(p + e for p, e in zip(prefix, extend)),
    ]
    assert backend._flash_attn_prefill_max_seq_len == 7
    assert backend._flash_attn_prefill_max_k_seq_len == max(
        p + e for p, e in zip(prefix, extend)
    )
    assert batch.extend_seq_lens.tolist() == [7, 5]
    q = torch.ones(12, 128, dtype=torch.bfloat16)
    k = torch.ones(12, 1, 64, dtype=torch.bfloat16)
    actual = backend.forward_extend(q, k, k, make_layer(), batch)
    assert actual.shape == q.shape
    assert backend.token_to_kv_pool.writes == 1


@pytest.mark.parametrize("prefixes", [(0, 0), (63, 65)])
@pytest.mark.parametrize("mask_mode", ["FULL_MASK", "QLEN_ONLY"])
def test_supplied_linear_chain_mask_is_validated_and_accepted(
    metadata_module, prefixes, mask_mode
):
    width = 4
    pieces = []
    for prefix in prefixes:
        columns = prefix + width if mask_mode == "FULL_MASK" else width
        query = torch.arange(width)[:, None]
        key = torch.arange(columns)[None, :]
        pieces.append(
            (key <= query + (prefix if mask_mode == "FULL_MASK" else 0)).flatten()
        )
    mask = torch.cat(pieces)
    original = mask.clone()
    metadata_module.validate_linear_verify_mask(
        mask, prefixes, width, mask_mode=SimpleNamespace(name=mask_mode)
    )
    torch.testing.assert_close(mask, original)


def test_packed_mask_is_rejected_before_vendor_layout_conversion(metadata_module):
    with pytest.raises(RuntimeError, match="boolean|packed|mask"):
        metadata_module.validate_linear_verify_mask(
            torch.ones(32, dtype=torch.uint8), [63, 65], 4
        )


def test_first_multistep_draft_initializes_children_before_selecting_write_slice(
    backend_module, monkeypatch
):
    # Upstream TritonMultiStepDraftBackend prepares every child's metadata
    # before EagleWorker selects that step's out_cache_loc slice for forward.
    children = [make_mtp_backend(backend_module, monkeypatch) for _ in range(2)]
    children[1].token_to_kv_pool = children[0].token_to_kv_pool
    children[1].req_to_token = children[0].req_to_token
    pool = children[0].token_to_kv_pool
    prefixes, num_steps = [63, 65], 3
    batch = EagerBatchView(**vars(make_spec_batch(children[0], prefixes)))
    batch.forward_mode = SpecMode.DECODE
    full_locations = torch.stack(
        [
            children[0].req_to_token[row, prefix : prefix + num_steps]
            for row, prefix in enumerate(prefixes)
        ]
    )
    batch.out_cache_loc = full_locations.flatten()
    expected_writes, expected_lengths = [], []
    batch.spec_info = SimpleNamespace()
    for step, child in enumerate(children):
        lengths = [prefix + step + 1 for prefix in prefixes]
        batch.spec_info.kv_indptr = torch.tensor(
            [0, lengths[0], sum(lengths)], dtype=torch.int32
        )
        batch.spec_info.kv_indices = torch.cat(
            [child.req_to_token[row, :length] for row, length in enumerate(lengths)]
        )
        child.init_forward_metadata(batch)
        expected_lengths.append(lengths)
        expected_writes.append(full_locations[:, step].clone())
        # All B*steps locations still exist while children are initialized.
        assert batch.out_cache_loc.numel() == 6
        torch.testing.assert_close(
            child._flash_attn_speculative_decode_writes, expected_writes[step]
        )
        assert child._flash_attn_forward_seq_lens.tolist() == lengths
    assert (
        children[0]._flash_attn_speculative_decode_writes.data_ptr()
        != children[1]._flash_attn_speculative_decode_writes.data_ptr()
    )
    assert (
        children[0]._flash_attn_forward_page_table.data_ptr()
        != children[1]._flash_attn_forward_page_table.data_ptr()
    )
    # Child1 crosses the first request's page boundary. It must not overwrite
    # child0's captured eager page view or overwrite the saved per-step tails.
    assert children[0]._flash_attn_forward_page_table[0, 1].item() == -1
    assert children[1]._flash_attn_forward_page_table[0, 1].item() == 2
    calls = []
    for step, child in enumerate(children):

        def vendor(*, _step=step, **kwargs):
            calls.append(_step)
            assert kwargs["cache_seqlens"].tolist() == expected_lengths[_step]
            torch.testing.assert_close(
                pool.k[expected_writes[_step]],
                torch.full((2, 1, 64), _step + 1, dtype=torch.bfloat16),
            )
            return torch.ones_like(kwargs["q"])

        child._flash_attn_with_kvcache = vendor
    execution_batch = clone_eager_batch(batch)
    q = torch.ones(2, 128, dtype=torch.bfloat16)
    k = torch.ones(2, 1, 64, dtype=torch.bfloat16)
    # Initializing with all locations is allowed; forwarding all locations is
    # not. An invalid slice must fail before any cache write/vendor call.
    with pytest.raises(RuntimeError, match="per-step KV tails"):
        children[0].forward_decode(q, k, k, make_layer(), execution_batch)
    assert pool.writes == 0 and calls == []
    execution_batch.out_cache_loc = full_locations[:, 0].contiguous()
    children[0].forward_decode(q, k, k, make_layer(), execution_batch)
    assert pool.writes == 1 and calls == [0]
    # The same SpecInput now holds last-child KV indices, while child0 used
    # its own saved per-step metadata on the cloned eager buffer view. Child1 must reject child0's stale write slice.
    with pytest.raises(RuntimeError, match="per-step KV tails"):
        children[1].forward_decode(q, k * 2, k * 2, make_layer(), execution_batch)
    assert pool.writes == 1 and calls == [0]
    execution_batch.out_cache_loc = full_locations[:, 1].flip(0)
    with pytest.raises(RuntimeError, match="per-step KV tails"):
        children[1].forward_decode(q, k * 2, k * 2, make_layer(), execution_batch)
    assert pool.writes == 1 and calls == [0]
    execution_batch.out_cache_loc = full_locations[:, 1].contiguous()
    children[1].forward_decode(q, k * 2, k * 2, make_layer(), execution_batch)
    assert pool.writes == 2 and calls == [0, 1]


def test_draft_decode_cannot_skip_saving_its_current_kv_token(
    backend_module, monkeypatch
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    batch = make_spec_batch(backend)
    batch.forward_mode = SpecMode.DECODE
    lengths = [64, 66]
    batch.spec_info = SimpleNamespace(
        kv_indptr=torch.tensor([0, 64, 130], dtype=torch.int32),
        kv_indices=torch.cat(
            [backend.req_to_token[row, :length] for row, length in enumerate(lengths)]
        ),
    )
    batch.out_cache_loc = torch.stack(
        [backend.req_to_token[row, length - 1] for row, length in enumerate(lengths)]
    )
    backend.init_forward_metadata(batch)
    calls = []
    backend._flash_attn_with_kvcache = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(RuntimeError, match="requires saving"):
        backend.forward_decode(
            torch.ones(2, 128, dtype=torch.bfloat16),
            None,
            None,
            make_layer(),
            batch,
            save_kv_cache=False,
        )
    assert backend.token_to_kv_pool.writes == 0
    assert calls == []


def test_speculative_overlap_is_rejected_but_ordinary_attention_is_unchanged(
    backend_module, monkeypatch
):
    monkeypatch.setattr(backend_module, "ForwardMode", SpecMode)
    runner = make_runner(decode="disabled")
    runner.server_args.disable_overlap_schedule = False
    # Ordinary non-speculative HCU does not acquire the new MTP-only guard.
    ordinary = backend_module.HCUAttnBackend(runner)
    assert not ordinary._flash_attn_speculative_enabled
    runner.server_args.speculative_algorithm = "EAGLE"
    runner.server_args.speculative_eagle_topk = 1
    runner.server_args.speculative_num_steps = 3
    runner.server_args.speculative_num_draft_tokens = 4
    with pytest.raises(RuntimeError, match="disable_overlap_schedule=True"):
        backend_module.HCUAttnBackend(runner)


@dataclass
class EagerBatchView:
    """Only slot/non-slot fields relevant to the pinned eager registry contract."""

    batch_size: int
    forward_mode: object
    global_forward_mode: object
    spec_info: object
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_sum: int
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    extend_prefix_lens_cpu: object
    extend_seq_lens_cpu: object
    extend_seq_lens: object
    encoder_lens: object


def clone_eager_batch(batch):
    # registry.extract_buffer uses dataclasses.replace, retaining non-slot
    # SpecInput identity but supplying new seq/req/out tensor buffer views.
    return replace(
        batch,
        seq_lens=batch.seq_lens.clone(),
        seq_lens_cpu=batch.seq_lens_cpu.clone(),
        req_pool_indices=batch.req_pool_indices.clone(),
        out_cache_loc=batch.out_cache_loc.clone(),
    )


def test_eager_registry_cloned_verify_view_preserves_metadata_contract(
    backend_module, monkeypatch
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    live = EagerBatchView(**vars(make_spec_batch(backend)))
    backend.init_forward_metadata(live)
    view = clone_eager_batch(live)
    assert view is not live and view.spec_info is live.spec_info
    for name in ("seq_lens", "req_pool_indices", "out_cache_loc"):
        assert getattr(view, name).data_ptr() != getattr(live, name).data_ptr()
    # Authentication must use immutable value snapshots, not aliases of the
    # scheduler's original tensors after the eager runner has copied them.
    live.seq_lens.add_(8)
    q = torch.ones(8, 128, dtype=torch.bfloat16)
    k = torch.ones(8, 1, 64, dtype=torch.bfloat16)
    actual = backend.forward_extend(q, k, k, make_layer(), view)
    assert actual.shape == q.shape
    assert backend.token_to_kv_pool.writes == 1
    torch.testing.assert_close(backend.token_to_kv_pool.k[view.out_cache_loc], k)


@pytest.mark.parametrize("changed", ["spec", "mode", "batch", "seq", "req", "out"])
def test_eager_registry_changed_verify_contract_fails_before_cache_write(
    backend_module, monkeypatch, changed
):
    backend = make_mtp_backend(backend_module, monkeypatch)
    live = EagerBatchView(**vars(make_spec_batch(backend)))
    backend.init_forward_metadata(live)
    view = clone_eager_batch(live)
    if changed == "spec":
        view.spec_info = SimpleNamespace(**vars(view.spec_info))
    elif changed == "mode":
        view.forward_mode = SpecMode.EXTEND
    elif changed == "batch":
        view.batch_size = 1
    elif changed == "seq":
        view.seq_lens[0] += 1
    elif changed == "req":
        view.req_pool_indices = view.req_pool_indices.flip(0)
    else:
        view.out_cache_loc[0] += 1
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["q"])

    backend._flash_attn_varlen_func = vendor
    with pytest.raises(RuntimeError, match="speculative|MTP|writes|metadata"):
        backend.forward_extend(
            torch.ones(8, 128, dtype=torch.bfloat16),
            torch.ones(8, 1, 64, dtype=torch.bfloat16),
            torch.ones(8, 1, 64, dtype=torch.bfloat16),
            make_layer(),
            view,
        )
    assert backend.token_to_kv_pool.writes == 0
    assert calls == []
