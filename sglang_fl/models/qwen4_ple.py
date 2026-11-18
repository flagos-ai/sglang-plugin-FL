# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen3.8 PLE orchestration using the shared operator dispatcher."""

import torch
from sglang_fl.dispatch import resolve_op


def _forward(original, self, hidden_states, forward_batch, batch):
    if not getattr(self, "_fl_qwen38", False) or batch.mode.is_target_verify():
        return original(self, hidden_states, forward_batch, batch)
    from sglang.srt.models.qwen4_exp import _pad_token_rows

    gate_norm = resolve_op("ple_gate_norm")
    hidden_states = hidden_states[: batch.processed_tokens]
    if self._prefetch_state is not None:
        embeddings = self._consume_prefetched_embeddings(forward_batch)
    else:
        embeddings = self.ple_embedding(batch, forward_batch)
    key, _ = self.key_proj(embeddings)
    value, _ = self.value_proj(embeddings)
    rows = hidden_states.shape[0]
    key = key.reshape(rows, self.hc_count, self.hidden_size)
    query = hidden_states.reshape(rows, self.hc_count, self.hidden_size)
    gated, normalized = gate_norm(
        key,
        query,
        value,
        self.norm_key.weight,
        self.norm_query.weight,
        self.norm_conv.weight,
        self.hc_count,
        self.norm_key.eps,
    )
    output = gated.flatten(-2) + self._short_conv(
        normalized.flatten(-2), forward_batch, batch
    )
    if not batch.use_decode_fast_path:
        output = torch.where(
            batch.valid_tokens.unsqueeze(-1), output, torch.zeros_like(output)
        )
    return _pad_token_rows(output, batch.physical_tokens)


def _short_conv(original, self, x, forward_batch, batch):
    from sglang.srt.models.qwen4_exp import (
        SHORT_CONV_PART,
        _ple_track_targets,
        get_token_to_kv_pool,
    )

    if (
        not getattr(self, "_fl_qwen38", False)
        or batch.mode.is_target_verify()
        or _ple_track_targets(forward_batch, batch) is not None
    ):
        return original(self, x, forward_batch, batch)
    convolution = resolve_op("ple_short_conv_")
    if not x.shape[0]:
        return x
    view = get_token_to_kv_pool().layer_router.read(self.layer_id, SHORT_CONV_PART)
    state = view.view.layer_cache(view.view_layer_id)
    output = torch.empty_like(x)
    starts = None if batch.use_decode_fast_path else batch.query_start_loc
    convolution(
        x,
        output,
        state,
        self.conv1d.weight.squeeze(1),
        batch.state_indices,
        dilation=self.short_conv_dilation,
        query_start_loc=starts,
        null_block_id=0,
    )
    return output


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    prefix = "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer."
    return [
        (prefix + "forward", _forward, HookType.AROUND),
        (prefix + "_short_conv", _short_conv, HookType.AROUND),
    ]
