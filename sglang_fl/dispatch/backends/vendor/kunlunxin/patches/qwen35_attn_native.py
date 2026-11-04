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

"""Kunlunxin: force Qwen3.5/3.6 full-attn prepare onto native torch path.

SGLang 0.5.18 added the CUDA Triton fused path ``fused_qk_gemma_rmsnorm_rope_gate``.
On cuda-compat, ``_is_cuda`` is True so ``Qwen3_5AttentionDecoderLayer.self_attention``
selects ``forward_prepare_cuda_fused`` (that Triton kernel) plus ``fused_sigmoid_mul``.
Those NVIDIA/Hopper-oriented kernels poison hidden states on Kunlun (first
full-attn layer → NaN into subsequent GDN layers → all-``!`` logits). Route to
``forward_prepare_native`` + torch sigmoid*gate.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
_applied = False


def _native_self_attention(self, positions, hidden_states, forward_batch):
    q, k, v, gate = self.forward_prepare_native(
        positions=positions,
        hidden_states=hidden_states,
    )
    attn_output = self.attn(q, k, v, forward_batch)
    if self.attn_output_gate and gate is not None:
        if gate.ndim == 3:
            gate = gate.reshape(gate.shape[0], -1)
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = self.o_proj(attn_output)
    return output


def patch_qwen35_attn_native() -> None:
    """Rebind full-attn prepare/gate fusion to torch-native on klx."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
    except Exception as e:
        logger.warning("qwen35_attn_native: cannot import qwen3_5: %s", e)
        return

    Qwen3_5AttentionDecoderLayer.self_attention = _native_self_attention
    logger.info(
        "patched Qwen3_5AttentionDecoderLayer.self_attention -> native "
        "QK-norm/RoPE/gate (avoids Triton fused_qk_gemma_rmsnorm_rope_gate "
        "on Kunlunxin)"
    )
