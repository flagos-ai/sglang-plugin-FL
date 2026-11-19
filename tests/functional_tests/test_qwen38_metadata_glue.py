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
"""Exercise the framework metadata glue graph through the QSA dispatch hook."""

from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_musa")
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (  # noqa: E402
    QwenSparseAttnBackend,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.model_executor.runner.metadata_glue_graph import (  # noqa: E402
    MetadataGlueGraph,
)
from sglang_fl.dispatch import resolve_op  # noqa: E402
from sglang_fl.models import qwen4_metadata  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.musa.is_available(), reason="requires MUSA graph capture and replay"
)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_metadata_glue_updates_inputs_and_rebuilds_replaced_buffers(rows):
    import flaggems_sglang
    from flaggems_sglang.reference import get_reference

    operation = resolve_op("qsa_build_decode_metadata")
    assert operation is flaggems_sglang.get_op("qsa_build_decode_metadata")
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    backend.token_to_kv_pool.qsa_compressed_page_size = 16
    calls = []

    def replay(self, **kwargs):
        calls.append(kwargs["bs"])
        return qwen4_metadata._replay_metadata(None, self, **kwargs)

    backend._replay_cuda_graph_metadata = MethodType(replay, backend)
    glue = MetadataGlueGraph("musa", strict=True)
    key = (rows, str(ForwardMode.DECODE), str(ForwardMode.DECODE))
    previous_table = None
    previous_outputs = None
    for generation in range(2):
        glue.reset()
        backend.req_to_token = torch.arange(
            (rows + 1) * 8192, device="musa", dtype=torch.int32
        ).reshape(rows + 1, 8192)
        one = lambda: torch.full((rows,), -99, device="musa", dtype=torch.int32)
        outputs = [one() for _ in range(5)]
        outputs += [
            torch.full((rows, 129), -99, device="musa", dtype=torch.int32),
            one(),
            torch.full((rows,), -99, device="musa", dtype=torch.int64),
            torch.full((rows, 4), -99, device="musa", dtype=torch.int32),
        ]
        if previous_table is not None:
            assert previous_table.data_ptr() != backend.req_to_token.data_ptr()
            assert previous_outputs[0].data_ptr() != outputs[0].data_ptr()
        indexer = SimpleNamespace(
            graph_prefix_lengths=outputs[2],
            graph_compressed_lengths=outputs[3],
            graph_write_locs=outputs[4],
            graph_compressed_page_table=outputs[5],
            decode_logical_positions=outputs[6],
            pending_ring_slots=outputs[7],
            graph_ring_group_locs=outputs[8],
            compress_ratio=4,
            token_to_kv_pool=backend.token_to_kv_pool,
        )
        metadata = SimpleNamespace(
            sequence_lengths=outputs[0],
            row_req_pool_indices=outputs[1],
            indexer_metadata=indexer,
        )
        backend._cuda_graph_metadata = {(ForwardMode.DECODE, rows): metadata}
        batch = SimpleNamespace(
            batch_size=rows,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            req_pool_indices=torch.arange(
                1, rows + 1, device="musa", dtype=torch.int32
            ),
            seq_lens=torch.full((rows,), 1024, device="musa", dtype=torch.int32),
            seq_lens_cpu=None,
            num_padding=0,
        )
        for _ in range(MetadataGlueGraph.NUM_WARMUP + 1):
            glue.run(backend, batch, key)
        assert glue.is_captured(key)
        captured_graph = glue._states[key]["graph"]
        capture_calls = len(calls)
        for phase in range(3):
            lengths = ([1024, 2048, 2049, 8192, 1, 4, 5] * rows)[phase : phase + rows]
            batch.seq_lens.copy_(torch.tensor(lengths, device="musa"))
            batch.req_pool_indices.copy_(batch.req_pool_indices.roll(1))
            backend.req_to_token.add_(64)
            backend.req_to_token[0].zero_()
            for output in outputs:
                output.fill_(-123)
            args = [
                batch.seq_lens,
                batch.req_pool_indices,
                backend.req_to_token,
                *outputs,
                4,
                64,
                0,
            ]
            expected = [
                x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in args
            ]
            get_reference("qsa_build_decode_metadata")(*expected)
            backend.forward_metadata = None
            glue.run(backend, batch, key)
            torch.musa.synchronize()
            assert len(calls) == capture_calls
            assert glue._states[key]["graph"] is captured_graph
            assert backend.forward_metadata is metadata
            for output, reference in zip(outputs, expected[3:12]):
                torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0)
        previous_table, previous_outputs = backend.req_to_token, outputs
    glue.reset()
