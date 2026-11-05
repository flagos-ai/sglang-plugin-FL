# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 adapter for the FlagGems INT8 indexer cache and logits."""

from types import MethodType

import torch

try:
    from flag_gems.fused.int8_sparse_indexer import (
        int8_indexer_cache_store,
        int8_indexer_paged_logits,
        int8_indexer_quantize,
    )
except ImportError as exc:
    raise ImportError(
        "Hy4-preview INT8 indexing requires the FlagGems int8_sparse_indexer "
        "quantize/cache-store/paged-logits API build; stock FlagGems alone "
        "does not provide this adapter dependency"
    ) from exc
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _page_table_rows(metadata, table, lengths, device):
    """Return ONLY the physical-table row map; keep request metadata compact.

    Native TARGET_VERIFY/DRAFT_EXTEND_V2 repeats its real page table per query,
    while cache_seqlens stays [B]. Ordinary extend instead supplies the compact
    [B, pages] table plus token_to_batch_idx. These index spaces are distinct.
    All branching below uses static shape facts, never captured tensor values.
    """
    rows = metadata.get_token_to_batch_idx()
    if rows is not None:
        return rows
    batch = metadata.get_seqlens_int32().numel()
    count = lengths.numel()
    if table.shape[0] == count:
        return torch.arange(count, device=device, dtype=torch.int32)
    if batch > 0 and table.shape[0] == batch and (count % batch == 0):
        return torch.arange(count, device=device, dtype=torch.int32) // (count // batch)
    raise ValueError(
        "HY4 INT8 table rows must be expanded per query or compact per request"
    )


def _forward_int8_core(
    self,
    x,
    q_lora,
    positions,
    forward_batch,
    layer_id,
    return_indices=True,
    *,
    out_cache_loc=None,
):
    if self.dsa_enable_prefill_cp or forward_batch.attn_cp_metadata is not None:
        raise ValueError(
            "HYV4 INT8 indexer currently supports TP/PP without context parallelism"
        )
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    if metadata is None:
        return None
    query, key, weights_raw = self._get_q_k_bf16(
        q_lora, x, positions, False, forward_batch=forward_batch
    )
    q, scale = int8_indexer_quantize(query)
    pool = get_token_to_kv_pool()
    cache = pool.get_index_k_with_scale_buffer(layer_id)
    slots = forward_batch.out_cache_loc if out_cache_loc is None else out_cache_loc
    int8_indexer_cache_store(key, cache, slots, pool.page_size)
    if not return_indices:
        return None
    if weights_raw is None:
        weights_raw = self._weights_proj_bf16_in_fp32_out(x)
    weights = (
        weights_raw.float()
        * scale.squeeze(-1)
        * self.softmax_scale
        * self.n_heads ** (-0.5)
    )
    table = metadata.get_page_table_64()
    lengths = metadata.get_seqlens_expanded()
    if metadata.get_seqlens_int32().numel() == 0:
        return torch.full(
            (q.shape[0], self.index_topk), -1, dtype=torch.int32, device=q.device
        )
    rows = _page_table_rows(metadata, table, lengths, q.device)
    count = rows.numel()
    logits = int8_indexer_paged_logits(
        q[:count],
        cache,
        weights[:count],
        table,
        rows.contiguous(),
        lengths[:count].contiguous(),
        table.shape[1] * pool.page_size,
        pool.page_size,
    )
    self._mask_init_and_local_tokens(logits, lengths[:count])
    topk = metadata.topk_transform(logits, self.index_topk)
    if count != q.shape[0]:
        padded = torch.full(
            (q.shape[0], self.index_topk), -1, device=q.device, dtype=topk.dtype
        )
        padded[:count] = topk
        topk = padded
    return topk


@eager_on_graph(True)
def _bcg_int8_prefill(layer_id, x, q_lora, positions, topk_result):
    context = get_tc_piecewise_forward_context()
    batch = context.forward_batch
    count = batch.extend_num_tokens
    result = _forward_int8_core(
        context.dsa_indexers[layer_id],
        x[:count],
        q_lora[:count],
        positions[..., :count],
        batch,
        layer_id,
        topk_result.numel() != 0,
        out_cache_loc=batch.out_cache_loc[:count],
    )
    if topk_result.numel() != 0:
        topk_result.fill_(-1)
        topk_result[: result.shape[0]].copy_(result)


def _forward_int8(
    self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
):
    if is_in_breakable_cuda_graph():
        shape = (x.shape[0] if return_indices else 0, self.index_topk)
        result = torch.full(shape, -1, device=x.device, dtype=torch.int32)
        _bcg_int8_prefill(layer_id, x, q_lora, positions, result)
        return result if return_indices else None
    return _forward_int8_core(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices
    )


def install_int8_indexer(indexer):
    indexer.forward = MethodType(_forward_int8, indexer)
