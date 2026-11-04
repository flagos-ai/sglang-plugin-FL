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

"""Kunlunxin: replace Triton ``fused_qkv_split_gdn_prefill`` with torch.split.

SGLang 0.5.18 added the CUDA Triton fused path ``fused_qkv_split_gdn_prefill``.
On cuda-compat Kunlun, ``gdn_backend`` uses it whenever ``qkv_dim <= 8192``.
That kernel silently corrupts K/V on strided post-conv views (V abs values
~1e37 → NaN in ``klx_gated_delta_net``).

TP1 for Qwen3.6-27B has ``qkv_dim = 10240 > 8192`` so it already takes the
``torch.split`` fallback and is healthy. TP2/TP4 drop to 5120/2560 and hit the
broken fused path → all-``!`` logits. Force the torch.split fallback on Kunlun.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
_applied = False


def _torch_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    """Match ``gdn_backend``'s non-fused branch (contiguous FLA layout)."""
    seq_len = mixed_qkv.shape[0]
    query, key, value = torch.split(
        mixed_qkv,
        [num_q_heads * head_q, num_k_heads * head_k, num_v_heads * head_v],
        dim=-1,
    )
    query = query.view(1, seq_len, num_q_heads, head_q)
    key = key.view(1, seq_len, num_k_heads, head_k)
    value = value.view(1, seq_len, num_v_heads, head_v)
    return query, key, value


def patch_gdn_fused_qkv_split() -> None:
    """Rebind gdn_backend fused QKV split to torch.split on Kunlunxin."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from sglang.srt.layers.attention.linear import gdn_backend
    except Exception as e:
        logger.warning("gdn_fused_qkv_split: cannot import gdn_backend: %s", e)
        return

    if not hasattr(gdn_backend, "fused_qkv_split_gdn_prefill"):
        logger.warning(
            "gdn_fused_qkv_split: gdn_backend has no fused_qkv_split_gdn_prefill"
        )
        return

    gdn_backend.fused_qkv_split_gdn_prefill = _torch_qkv_split_gdn_prefill
    # Also force the call-site gate so even a stale import of the Triton symbol
    # cannot be selected (qkv_dim <= 8192 is true under TP for Qwen3.6-27B).
    if hasattr(gdn_backend, "MAX_FUSED_QKV_SPLIT_DIM"):
        gdn_backend.MAX_FUSED_QKV_SPLIT_DIM = 0
    logger.warning(
        "patched gdn_backend.fused_qkv_split_gdn_prefill -> torch.split "
        "and MAX_FUSED_QKV_SPLIT_DIM=0 "
        "(avoids Triton fused QKV split corruption on Kunlunxin TP)"
    )
