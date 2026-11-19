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
"""Flatten framework-owned QSA decode buffers for the unified dispatch API."""

from sglang_fl.dispatch import resolve_op


def _replay_metadata(
    original,
    self,
    *,
    bs,
    req_pool_indices,
    seq_lens,
    forward_mode,
    spec_info,
    seq_lens_cpu,
    num_padding=0,
):
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    if (
        seq_lens.device.type != "musa"
        or not forward_mode.is_decode()
        or spec_info is not None
        or self.req_to_token is None
        or not isinstance(self.token_to_kv_pool, QSATokenToKVPool)
    ):
        return original(
            self,
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
            num_padding=num_padding,
        )
    metadata = self._cuda_graph_metadata[(forward_mode, bs)]
    indexer = metadata.indexer_metadata
    operation = resolve_op("qsa_build_decode_metadata")
    operation(
        seq_lens,
        req_pool_indices,
        self.req_to_token,
        metadata.sequence_lengths,
        metadata.row_req_pool_indices,
        indexer.graph_prefix_lengths,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.graph_compressed_page_table,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
        indexer.compress_ratio,
        indexer.token_to_kv_pool.qsa_compressed_page_size * indexer.compress_ratio,
        num_padding or 0,
    )
    self.forward_metadata = metadata


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.attention.qwen_sparse_attn_backend."
            "QwenSparseAttnBackend._replay_cuda_graph_metadata",
            _replay_metadata,
            HookType.AROUND,
        )
    ]
