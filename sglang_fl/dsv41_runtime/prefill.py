# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (Apache-2.0).
"""DeepSeek V4.1 candidate selection without duplicating full score chunks."""

import torch
import torch.nn.functional as F


def _candidate_block_amax(logits, block_size):
    width = logits.size(-1)
    padding = -width % block_size
    if padding == 0:
        return logits.unflatten(-1, (-1, block_size)).amax(dim=-1)
    full_width = width - width % block_size
    tail = F.pad(logits[..., full_width:], (0, padding), value=-torch.inf)
    tail_scores = tail.amax(dim=-1, keepdim=True)
    if full_width == 0:
        return tail_scores
    full_scores = logits[..., :full_width].unflatten(-1, (-1, block_size)).amax(dim=-1)
    return torch.cat((full_scores, tail_scores), dim=-1)


def candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = _candidate_block_amax(logits, block_size)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def candidate_block_ids(logits, compress_lens, topk_blocks, block_size):
    scores = _candidate_block_amax(logits, block_size)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    ids = torch.where(top.values > -torch.inf, top.indices, num_blocks)
    return ids.to(torch.int32).sort(dim=-1).values


def extend_indexer(self, layer, x, q_lora, pos, forward_batch) -> None:
    """Hopper FP4 indexer for ragged prefill rows grouped by request."""
    from sglang.srt.layers.attention import deepseek_v4_backend as native

    pool = self.token_to_kv_pool
    core = self.forward_metadata.core_metadata
    ratio = layer.compress_ratio
    indexer = layer.indexer
    page_indices = core.sparse_page_indices(ratio)
    raw_indices = core.sparse_raw_indices(ratio)
    page_indices.fill_(-1)
    if raw_indices is not None:
        raw_indices.fill_(-1)
    tail = self.forward_metadata.late_layer_tail
    q_lens_cpu = (
        tail.extend_seq_lens_cpu
        if tail is not None
        else native._as_int_list(forward_batch.extend_seq_lens_cpu)
    )
    seq_lens_cpu = native._as_int_list(forward_batch.seq_lens_cpu)
    assert q_lens_cpu is not None and seq_lens_cpu is not None
    assert len(q_lens_cpu) == len(seq_lens_cpu)
    live_tokens = sum(q_lens_cpu)
    if any((t.shape[0] < live_tokens for t in (x, q_lora, pos))):
        raise ValueError("Hopper EXTEND indexer cannot truncate live query rows")
    x, q_lora, pos = (x[:live_tokens], q_lora[:live_tokens], pos[:live_tokens])
    q = indexer.queries(q_lora, layer.freqs_cis[pos])
    weights = indexer.head_weights(x)
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    topk_offsets = torch.zeros_like(compress_lens)
    table = pool.get_index_k_with_scale_buffer(layer.layer_id)
    page_size = table.shape[1] // 68
    req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
    topk = indexer.index_topk
    publish = [] if indexer.is_candidate_source else None
    consume = self.prefill_candidate_blocks if indexer.uses_candidates else None
    empty_mask = torch.empty(
        0,
        0,
        dtype=torch.int32,
        device=pos.device,
    )
    row_base = 0
    for b, (q_len, seq_len) in enumerate(zip(q_lens_cpu, seq_lens_cpu)):
        lc = seq_len // ratio
        if lc == 0 or q_len == 0:
            if publish is not None:
                publish.append(empty_mask)
            row_base += q_len
            continue
        j = torch.arange(lc, device=pos.device)
        req_row = torch.index_select(
            self.req_to_token, 0, req_pool_indices[b : b + 1]
        ).squeeze(0)
        slots_j = req_row[j * ratio].to(torch.int64) // ratio
        index_k = native.unpack_fp4_index_keys_to_fp8(slots_j, table, page_size)
        k = min(topk, lc)
        score_width = (
            consume[b].shape[1] * indexer.candidate_block_size
            if consume is not None
            else lc
        )
        rows_per_chunk = max(
            1, native._TORCH_INDEXER_SCORE_BUDGET_BYTES // (score_width * 4)
        )
        masks = [] if publish is not None else None
        for start in range(0, q_len, rows_per_chunk):
            stop = min(q_len, start + rows_per_chunk)
            token_rows = slice(row_base + start, row_base + stop)
            local_rows = slice(start, stop)
            lens_c = compress_lens[token_rows]
            q_fp8 = native.quantize_bf16_index_queries_fp8(q[token_rows])
            block_ids = consume[b][local_rows] if consume is not None else None
            scores = native.fp8_index_logits_prefill(
                q_fp8,
                weights[token_rows],
                index_k,
                lens_c,
                candidate_blocks=block_ids,
                candidate_block_size=indexer.candidate_block_size,
            )
            if masks is not None:
                masks.append(
                    native.select_candidate_block_ids(
                        scores,
                        lens_c[:, None],
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    )
                )
            idx = torch.empty(
                (scores.shape[0], k), dtype=torch.int32, device=scores.device
            )
            score_lens = (
                torch.full_like(lens_c, scores.shape[1])
                if block_ids is not None
                else lens_c
            )
            native.topk_transform_ragged_v2(
                scores,
                score_lens,
                out_offsets=topk_offsets[token_rows],
                out_indices=idx,
            )
            if consume is not None and masks is None:
                idx = native._mask_topk_scores(scores, idx)
                if block_ids is not None:
                    selected_blocks = block_ids.gather(
                        1, (idx.clamp_min(0) // indexer.candidate_block_size).long()
                    )
                    logical = (
                        selected_blocks * indexer.candidate_block_size
                        + idx % indexer.candidate_block_size
                    )
                    idx = torch.where(idx >= 0, logical, -1)
            unselected = torch.iinfo(torch.int32).max
            idx = idx.masked_fill(idx < 0, unselected).sort(dim=-1).values
            reach = (idx != unselected) & (idx < lens_c[:, None])
            page_indices[token_rows, :k] = torch.where(
                reach, slots_j[idx.clamp_max(lc - 1)], -1
            ).to(torch.int32)
            if raw_indices is not None:
                raw_indices[token_rows, :k] = torch.where(reach, idx, -1).to(
                    torch.int32
                )
            # Every consumer has been enqueued on this stream. Drop this chunk
            # before the next allocation instead of retaining it during its RHS.
            del scores
        if masks is not None:
            publish.append(torch.cat(masks) if len(masks) > 1 else masks[0])
        row_base += q_len
    assert row_base == q.shape[0]
    if publish is not None:
        self.prefill_candidate_blocks = publish
        self.candidate_masks = None
