# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Share the QSA cache bound across layers and fuse eager partial RoPE."""

import torch
from sglang_fl.dispatch import resolve_op


def model_forward(original, self, input_ids, positions, forward_batch, *args, **kwargs):
    if not getattr(self, "_fl_qwen38", False):
        return original(self, input_ids, positions, forward_batch, *args, **kwargs)
    from sglang.srt.model_executor.runner import get_is_capture_mode

    old = getattr(forward_batch, "_fl_qsa_rope_bound", None)
    bound = None
    if (
        positions is not None
        and positions.numel()
        and positions.device.type == "musa"
        and not get_is_capture_mode()
    ):
        bound = int(positions.max().item())
    forward_batch._fl_qsa_rope_bound = bound
    try:
        return original(self, input_ids, positions, forward_batch, *args, **kwargs)
    finally:
        forward_batch._fl_qsa_rope_bound = old


def indexer_forward(
    original, self, hidden, positions, batch, metadata, *args, **kwargs
):
    if not getattr(self, "_fl_qwen38", False):
        return original(self, hidden, positions, batch, metadata, *args, **kwargs)
    bound = getattr(batch, "_fl_qsa_rope_bound", None)
    old = getattr(self, "_fl_qsa_rope_cache_ready", False)
    self._fl_qsa_rope_cache_ready = False
    if bound is not None and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length"):
        self.rotary_emb._ensure_cos_sin_cache_length(bound)
        self._fl_qsa_rope_cache_ready = True
    try:
        return original(self, hidden, positions, batch, metadata, *args, **kwargs)
    finally:
        self._fl_qsa_rope_cache_ready = old


def apply_rope(original, self, positions, tensor, ensure_cos_sin_cache=True):
    if not getattr(self, "_fl_qwen38", False):
        return original(
            self, positions, tensor, ensure_cos_sin_cache=ensure_cos_sin_cache
        )
    rope = self.rotary_emb
    cache = getattr(rope, "cos_sin_cache", None)
    if (
        tensor.device.type != "musa"
        or tensor.dtype not in (torch.bfloat16, torch.float16)
        or tensor.ndim != 3
        or positions.ndim not in (1, 2)
        or positions.shape[-1] != tensor.shape[0]
        or positions.device != tensor.device
        or positions.dtype not in (torch.int32, torch.int64)
        or (positions.ndim == 2 and positions.shape[0] != 3)
        or cache is None
        or cache.device != tensor.device
        or cache.ndim != 2
        or getattr(rope, "mrope_interleaved_glm", False)
        or (
            ensure_cos_sin_cache
            and not getattr(self, "_fl_qsa_rope_cache_ready", False)
        )
    ):
        return original(
            self, positions, tensor, ensure_cos_sin_cache=ensure_cos_sin_cache
        )
    out = resolve_op("qwen4_qsa_partial_rope")(
        tensor,
        positions,
        cache,
        self._rope_axis_map(tensor.device),
        rope.rotary_dim,
        rope.is_neox_style,
    )
    return out


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    prefix = "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer."
    return [
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpModel.forward",
            model_forward,
            HookType.AROUND,
        ),
        (prefix + "_forward_impl", indexer_forward, HookType.AROUND),
        (prefix + "apply_rope", apply_rope, HookType.AROUND),
    ]
