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

"""Kunlunxin: disable Triton fused vocab-parallel embedding under TP.

SGLang 0.5.18 added the CUDA Triton fused path ``fused_vocab_parallel_embedding``.
``VocabParallelEmbedding._use_triton_embedding`` is True whenever ``tp_size > 1``
and ``is_cuda()``. On cuda-compat Kunlun that kernel silently returns wrong
shard rows (TP ranks still agree after all-reduce, but the reconstructed
embedding diverges from TP1 → poor answer quality / early wrong EOS). Force
``_use_triton_embedding=False`` and the mask+gather+fill path instead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_vocab_embed_native() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    except Exception as e:
        logger.warning("vocab_embed_native: import failed: %s", e)
        return

    VocabParallelEmbedding._use_triton_embedding = lambda self, input_: False
    logger.warning(
        "patched VocabParallelEmbedding._use_triton_embedding -> False "
        "(avoids Triton fused vocab embedding corruption on Kunlunxin TP)"
    )
