# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen3.8 PLE orchestration using the shared operator dispatcher."""

import torch
from sglang_fl.dispatch import resolve_op


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
        (prefix + "_short_conv", _short_conv, HookType.AROUND),
    ]
