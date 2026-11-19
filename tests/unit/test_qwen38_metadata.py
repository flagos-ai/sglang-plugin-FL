# SPDX-License-Identifier: Apache-2.0
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
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang_fl.models import qwen4_metadata


def setup_case():
    outputs = [object() for _ in range(9)]
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    indexer = SimpleNamespace(
        graph_prefix_lengths=outputs[2],
        graph_compressed_lengths=outputs[3],
        graph_write_locs=outputs[4],
        graph_compressed_page_table=outputs[5],
        decode_logical_positions=outputs[6],
        pending_ring_slots=outputs[7],
        graph_ring_group_locs=outputs[8],
        compress_ratio=4,
        token_to_kv_pool=SimpleNamespace(qsa_compressed_page_size=16),
    )
    metadata = SimpleNamespace(
        sequence_lengths=outputs[0],
        row_req_pool_indices=outputs[1],
        indexer_metadata=indexer,
    )
    backend = SimpleNamespace(
        req_to_token=object(),
        token_to_kv_pool=pool,
        _cuda_graph_metadata={(ForwardMode.DECODE, 7): metadata},
        forward_metadata=None,
    )
    args = dict(
        bs=7,
        req_pool_indices=object(),
        seq_lens=SimpleNamespace(device=SimpleNamespace(type="musa", index=0)),
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=object(),
        num_padding=1,
    )
    return backend, metadata, outputs, args


def test_flat_dispatch_uses_framework_buffers(monkeypatch):
    backend, metadata, outputs, args = setup_case()
    operation = Mock()
    original = Mock()
    resolve = Mock(return_value=operation)
    monkeypatch.setattr(qwen4_metadata, "resolve_op", resolve)
    qwen4_metadata._replay_metadata(original, backend, **args)
    resolve.assert_called_once_with("qsa_build_decode_metadata")
    operation.assert_called_once_with(
        args["seq_lens"],
        args["req_pool_indices"],
        backend.req_to_token,
        *outputs,
        4,
        64,
        1,
    )
    assert backend.forward_metadata is metadata
    original.assert_not_called()


@pytest.mark.parametrize("case", ["cpu", "prefill", "spec", "pool"])
def test_unsupported_metadata_uses_original(monkeypatch, case):
    backend, metadata, outputs, args = setup_case()
    if case == "cpu":
        args["seq_lens"].device = torch.device("cpu")
    if case == "prefill":
        args["forward_mode"] = ForwardMode.EXTEND
    if case == "spec":
        args["spec_info"] = object()
    if case == "pool":
        backend.token_to_kv_pool = object()
    operation = Mock()
    monkeypatch.setattr(qwen4_metadata, "resolve_op", operation)
    original = Mock()
    qwen4_metadata._replay_metadata(original, backend, **args)
    original.assert_called_once_with(backend, **args)
    operation.assert_not_called()


def test_metadata_execution_failure_is_not_retried(monkeypatch):
    backend, metadata, outputs, args = setup_case()
    operation = Mock(side_effect=RuntimeError("kernel failed"))
    monkeypatch.setattr(qwen4_metadata, "resolve_op", lambda name: operation)
    original = Mock()
    with pytest.raises(RuntimeError, match="kernel failed"):
        qwen4_metadata._replay_metadata(original, backend, **args)
    assert operation.call_count == 1
    original.assert_not_called()
    assert backend.forward_metadata is None


def test_missing_metadata_capability_fails_before_mutation(monkeypatch):
    backend, metadata, outputs, args = setup_case()
    resolve = Mock(side_effect=RuntimeError("operator unavailable"))
    monkeypatch.setattr(qwen4_metadata, "resolve_op", resolve)
    original = Mock()
    with pytest.raises(RuntimeError, match="operator unavailable"):
        qwen4_metadata._replay_metadata(original, backend, **args)
    assert backend.forward_metadata is None
    original.assert_not_called()


def test_serving_does_not_capture_an_unknown_metadata_key():
    from sglang_fl.musa_graph import MusaDecodeGraphRunner

    runner = MusaDecodeGraphRunner.__new__(MusaDecodeGraphRunner)
    runner._metadata_glue = SimpleNamespace(is_captured=lambda key: False)
    runner._init_replay_attn_metadata_from_view = Mock()
    batch = object()
    runner._init_replay_attn_metadata(
        batch,
        raw_bs=7,
        bs=8,
        num_tokens=8,
        glue_key=("unknown",),
        early_staged=False,
        attention_variant=None,
    )
    runner._init_replay_attn_metadata_from_view.assert_called_once_with(
        batch,
        raw_bs=7,
        bs=8,
        num_tokens=8,
        glue_key=None,
        attention_variant=None,
    )
