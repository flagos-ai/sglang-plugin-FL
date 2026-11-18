# SPDX-License-Identifier: Apache-2.0
"""MUSA QSA dispatch and exact all-block selection for short contexts."""

import torch
from sglang_fl.dispatch import resolve_op


def _forward(original, self, hidden, positions, batch, metadata, *args, **kwargs):
    if not getattr(self, "_fl_qwen38", False):
        return original(self, hidden, positions, batch, metadata, *args, **kwargs)
    previous_metadata = getattr(self, "_fl_current_metadata", None)
    self._fl_current_metadata = metadata
    lengths = getattr(batch, "seq_lens_cpu", None)
    self._fl_all_decode_blocks = (
        lengths is not None
        and len(lengths) > 0
        and max(lengths) <= self.block_topk * self.compress_ratio
    )
    # A graph freezes this CPU decision. Record its validity bound so replay
    # cannot silently reuse the all-blocks path for a longer context.
    if getattr(metadata, "is_cuda_graph", False) and self._fl_all_decode_blocks:
        self._fl_graph_all_blocks_limit = self.block_topk * self.compress_ratio
    try:
        return original(self, hidden, positions, batch, metadata, *args, **kwargs)
    finally:
        self._fl_all_decode_blocks = False
        self._fl_current_metadata = previous_metadata


def _prefill(
    original, self, q, keys, starts, ends, positions, lengths, max_row_keys=None
):
    if not getattr(self, "_fl_qwen38", False) or max_row_keys is None:
        return original(self, q, keys, starts, ends, positions, lengths, max_row_keys)
    if max_row_keys > self.block_topk:
        from sglang.srt.layers.attention.qsa.qsa_indexer import (
            _qsa_prefill_row_chunk_size,
        )
        from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

        metadata = self._fl_current_metadata
        cache, pages, _, _ = metadata.get_decode_mqa_inputs(self.layer_id)
        request_ids = metadata.get_token_to_batch_idx()
        sequence_lengths = metadata.get_seqlens_int32()
        output = torch.empty(
            (q.shape[0], self.token_topk + self.compress_ratio - 1),
            dtype=torch.int32,
            device=q.device,
        )
        chunk = _qsa_prefill_row_chunk_size(q.shape[0], max_row_keys, q.shape[1])
        score = resolve_op("qsa_mqa_paged")
        select = resolve_op("qwen4_qsa_topk")
        for start in range(0, q.shape[0], chunk):
            rows = slice(start, min(start + chunk, q.shape[0]))
            logits, visible = score(
                q[rows],
                cache,
                pages,
                request_ids[rows],
                positions[rows],
                sequence_lengths,
                self.compress_ratio,
                num_columns=max_row_keys,
            )
            blocks = select(logits, torch.zeros_like(visible), visible, self.block_topk)
            output[rows] = expand_qsa_block_indices(
                blocks,
                positions[rows],
                lengths[rows],
                self.compress_ratio,
                self.token_topk,
            )
        return output
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    blocks = resolve_op("qwen4_qsa_all_blocks")(ends - starts, self.block_topk)
    return expand_qsa_block_indices(
        blocks, positions, lengths, self.compress_ratio, self.token_topk
    )


def _decode(
    original,
    self,
    q,
    cache,
    pages,
    compressed_lengths,
    max_model_len,
    positions,
    lengths,
    defer_expansion=False,
):
    if not getattr(self, "_fl_qwen38", False):
        return original(
            self,
            q,
            cache,
            pages,
            compressed_lengths,
            max_model_len,
            positions,
            lengths,
            defer_expansion,
        )
    if not getattr(self, "_fl_all_decode_blocks", False):
        from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

        requests = torch.arange(q.shape[0], dtype=torch.int32, device=q.device)
        logits, visible = resolve_op("qsa_mqa_paged")(
            q,
            cache,
            pages,
            requests,
            positions,
            lengths,
            self.compress_ratio,
            num_columns=max_model_len,
        )
        blocks = resolve_op("qwen4_qsa_topk")(
            logits, torch.zeros_like(visible), visible, self.block_topk
        )
        if defer_expansion:
            return blocks
        return expand_qsa_block_indices(
            blocks, positions, lengths, self.compress_ratio, self.token_topk
        )
    blocks = resolve_op("qwen4_qsa_all_blocks")(compressed_lengths, self.block_topk)
    if defer_expansion:
        return blocks
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    return expand_qsa_block_indices(
        blocks, positions, lengths, self.compress_ratio, self.token_topk
    )


def _no_cuda_jit(original, self, *args, **kwargs):
    if not getattr(self, "_fl_qwen38", False):
        return original(self, *args, **kwargs)
    return False


def _use_compress(original, self, pool):
    if not getattr(self, "_fl_qwen38", False):
        return original(self, pool)
    return (
        getattr(self, "_fl_qwen38", False)
        and self.index_head_dim == 128
        and self.rotary_emb.rotary_dim == 64
        and self.rotary_emb.is_neox_style
        and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
        and getattr(pool, "qsa_rope_position_buffer", None) is not None
    )


def _compress(
    original, self, pool, group_locs, write_locs, source_keys=None, source_rope=None
):
    if not getattr(self, "_fl_qwen38", False) or not _use_compress(None, self, pool):
        return original(self, pool, group_locs, write_locs, source_keys, source_rope)
    if source_keys is None:
        source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
    if source_rope is None:
        source_rope = pool.qsa_rope_position_buffer
    cache = pool.get_qsa_compressed_k_buffer(self.layer_id)
    rows = group_locs.shape[0]
    requests = torch.arange(rows, device=group_locs.device, dtype=torch.int32)
    positions = torch.full_like(requests, self.compress_ratio - 1)
    section = tuple(self.rotary_emb.mrope_section or (32, 0, 0))
    resolve_op("qsa_compress_norm_mrope_store_groups")(
        source_keys.view(-1, 1, 1, self.index_head_dim),
        group_locs,
        requests,
        positions,
        write_locs,
        cache.view(-1, 1, 1, self.index_head_dim),
        self.k_layernorm.weight,
        self.rotary_emb.cos_sin_cache,
        self.compress_ratio,
        self.k_layernorm.variance_epsilon,
        self.rotary_emb.rotary_dim,
        section,
        self.rotary_emb.mrope_interleaved,
        source_rope.view(-1, 1, 1, 3),
    )


def _attention(
    original,
    self,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache=True,
    topk_indices=None,
    **kwargs,
):
    if "qsa_output_gate" not in kwargs:
        return original(
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            topk_indices=topk_indices,
            **kwargs,
        )
    attention = resolve_op("qsa_sparse_paged_attention")
    store = resolve_op("qsa_store_kv_cache_rows")
    pool = self.token_to_kv_pool
    key_cache = pool.layer_router.read(layer.layer_id).key().unsqueeze(1)
    value_cache = pool.layer_router.read(layer.layer_id).value().unsqueeze(1)
    metadata = self._resolve_metadata(forward_batch)
    indices = self._expand_block_indices(topk_indices, metadata)
    if save_kv_cache:
        store(
            key_cache,
            value_cache,
            forward_batch.out_cache_loc,
            k.view(-1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, layer.tp_v_head_num, layer.head_dim),
        )
    query = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    num_rows = query.shape[0]
    valid_rows = indices.shape[0]
    gate = kwargs["qsa_output_gate"]
    if gate is not None:
        gate = gate.reshape(query.shape)[:valid_rows]
    output = attention(
        query[:valid_rows],
        key_cache,
        value_cache,
        indices,
        metadata.token_slot_table,
        metadata.token_to_batch_idx,
        layer.scaling,
        gate=gate,
    )
    return self._pad_extend_output(output, num_rows)


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    prefix = "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer."
    targets = []
    targets.extend(
        (prefix + name, function, HookType.AROUND)
        for name, function in (
            ("_use_fused_prep", _no_cuda_jit),
            ("_use_fused_compress", _use_compress),
            ("_fused_compress_store", _compress),
            ("_forward_impl", _forward),
            ("select_prefill_tokens", _prefill),
            ("select_decode_tokens", _decode),
        )
    )
    backend = (
        "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend."
    )
    targets.extend(
        (backend + name, _attention, HookType.AROUND)
        for name in ("forward_extend", "forward_decode")
    )
    targets.append(
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer.self_attention",
            _self_attention,
            HookType.AROUND,
        )
    )
    return targets


def _self_attention(
    original,
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: object,
) -> torch.Tensor:
    from sglang.srt.models import qwen4_exp as model

    if (
        not self.is_qsa
        or not getattr(self, "_fl_qwen38", False)
        or forward_batch.attn_cp_metadata is not None
        or forward_batch.forward_mode.is_target_verify()
    ):
        return original(self, positions, hidden_states, forward_batch)
    overlap_indexer = (
        self.is_qsa
        and self.alt_stream is not None
        and model.get_is_capture_mode()
        # Graph breaks must stay off the decode overlap stream.
        and not model.is_in_breakable_cuda_graph()
        and hidden_states.shape[0] < model._QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
    )
    attention_kwargs = {}
    if overlap_indexer:
        # Safe to overlap: the indexer reads only hidden_states/positions,
        # and writes QSA-private pool buffers.
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)

    q, k, v, gate = self._prepare_qkv_gate(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )

    if overlap_indexer:
        with torch.cuda.stream(self.alt_stream):
            topk_indices = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )
        current_stream.wait_stream(self.alt_stream)
        # Allocated on alt_stream, consumed by attention on the current
        # stream; tell the caching allocator before alt_stream is reused.
        topk_indices.record_stream(current_stream)
        attention_kwargs["topk_indices"] = topk_indices
    elif self.is_qsa:
        if (
            model._shards_rows_over_cp(forward_batch)
            and not model.get_is_capture_mode()
        ):
            from sglang.srt.layers.attention.qsa.glue import (
                resolve_qsa_sparse_backend,
            )

            # Every CP rank issues the K/V gather before the index-key
            # gather, overlapping it with the indexer's local computation.
            resolve_qsa_sparse_backend(model.get_attn_backend()).prefetch_cp_kv(
                self.attn, k, v, forward_batch
            )
        attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
            hidden_states, positions, forward_batch
        )

    attention_kwargs["qsa_output_gate"] = gate
    attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)
    output, _ = self.o_proj(attn_output)
    return output
