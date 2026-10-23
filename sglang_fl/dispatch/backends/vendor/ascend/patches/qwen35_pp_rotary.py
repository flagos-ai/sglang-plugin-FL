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

"""Refresh Qwen3.5/3.6 NPU RoPE positions on every pipeline stage."""

from __future__ import annotations

import inspect
from functools import wraps


_PATCH_MARKER = "_sglang_fl_pp_rotary_cache_patch"
_LOCAL_FIRST_MARKER = "_sglang_fl_pp_first_full_attention"
_EXPECTED_INIT_PARAMETERS = ("self", "config", "quant_config", "prefix", "is_nextn")
_EXPECTED_PREPARE_PARAMETERS = (
    "self",
    "positions",
    "hidden_states",
    "forward_batch",
)


def patch_qwen35_pp_rotary_cache() -> None:
    """Keep the native fused NPU attention path usable after PP partitioning.

    SGLang 0.5.18 initializes the shared position sin/cos tensors only at the
    model's *global* first full-attention layer. That layer is absent on later
    PP stages, so their fused NPU RoPE receives ``None`` during graph capture
    or decode. Mark the actual first local full-attention layer after the
    upstream model has applied its partition, including custom uneven splits.
    Refresh its cache for each call, not just when the cache is empty: decode
    positions change even when the batch size remains constant.

    The original initializer and fused preparation method remain responsible
    for all model construction and computation. The global first layer and
    PP=1 keep their existing single refresh. The MoE model inherits this base
    initializer, covering both Qwen3.6-27B and Qwen3.6-35B-A3B.
    """

    from sglang.srt.models.qwen3_5 import (
        Qwen3_5AttentionDecoderLayer,
        Qwen3_5ForCausalLM,
    )

    original_init = Qwen3_5ForCausalLM.__init__
    original_prepare = Qwen3_5AttentionDecoderLayer.forward_prepare_npu
    patched_init = getattr(original_init, _PATCH_MARKER, False)
    patched_prepare = getattr(original_prepare, _PATCH_MARKER, False)
    if patched_init and patched_prepare:
        return
    if patched_init or patched_prepare:
        raise RuntimeError("Incomplete SGLang Qwen3.5 Ascend PP RoPE patch")

    for original, expected in (
        (original_init, _EXPECTED_INIT_PARAMETERS),
        (original_prepare, _EXPECTED_PREPARE_PARAMETERS),
    ):
        actual = tuple(inspect.signature(original).parameters)
        if actual != expected:
            raise RuntimeError(
                "Unsupported SGLang Qwen3.5 interface for the Ascend PP RoPE "
                f"patch: {original.__qualname__}{actual}, expected {expected}."
            )

    @wraps(original_init)
    def init(self, config, quant_config=None, prefix="", is_nextn=False):
        original_init(
            self,
            config,
            quant_config=quant_config,
            prefix=prefix,
            is_nextn=is_nextn,
        )
        if self.pp_group.world_size <= 1:
            return

        # Do not infer the stage's range from rank or a uniform layer count.
        # make_layers() may honor a custom SGLANG_PP_LAYER_PARTITION instead.
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            if isinstance(layer, Qwen3_5AttentionDecoderLayer):
                # The upstream method already refreshes the global first one.
                if layer.attn.layer_id != config.full_attention_interval - 1:
                    setattr(layer, _LOCAL_FIRST_MARKER, True)
                break

    @wraps(original_prepare)
    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        if getattr(self, _LOCAL_FIRST_MARKER, False):
            self.rotary_emb.get_cos_sin_with_position(positions)
        return original_prepare(self, positions, hidden_states, forward_batch)

    setattr(init, _PATCH_MARKER, True)
    setattr(forward_prepare_npu, _PATCH_MARKER, True)
    Qwen3_5ForCausalLM.__init__ = init
    Qwen3_5AttentionDecoderLayer.forward_prepare_npu = forward_prepare_npu
