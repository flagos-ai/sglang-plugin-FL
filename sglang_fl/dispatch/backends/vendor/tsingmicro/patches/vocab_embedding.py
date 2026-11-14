# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep VocabParallelEmbedding on the torch path on txda (SGLang >= 0.5.18).

SGLang 0.5.18 added a fused Triton kernel for the TP-sharded embedding lookup
(``sglang.kernels.ops.embeddings.vocab_parallel_embedding``).  It is selected
whenever ``input_.is_cuda`` and ``tp_size > 1`` (only NPU is excluded), so on
txda -- where ``torch.cuda`` maps to torch_txda -- it replaces the torch
mask + gather + fill sequence.

On txda the kernel's masked weight load::

    tl.load(weight_ptr + local_id * weight_stride0 + cols,
            mask=col_mask & valid, other=0.0)

does not honor ``valid`` for tokens outside the local vocab shard: instead of
zeros it returns weight row 0.  Every token belongs to exactly one TP rank, so
after the embedding all_reduce each token gets ``tp_size - 1`` extra copies of
row 0 and the model output is garbage from the first token (Qwen3.6 35B TP8
answers only "\\n").  A single-card check (int32/int64 ids, several
BLOCK_H/num_warps) reproduces it: in-shard rows match exactly, every
out-of-shard row is non-zero.

``_use_triton_embedding`` is forced to ``False`` so SGLang takes its own torch
path (``get_masked_input_and_mask`` + embedding + ``masked_fill_``), identical
to SGLang 0.5.11.  Applied lazily, after SGLang imports the module.  Older
SGLang without the method: no-op.  Set ``SGLANG_FL_TXDA_TRITON_EMBEDDING=1`` to
keep the Triton kernel (e.g. to re-test after a flagtree fix).
"""

import os

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.kv_leak_fix import (
    _install_post_import_wrap,
)

_log = patch_logger("vocab_embedding")

_KEEP_KEY = "SGLANG_FL_TXDA_TRITON_EMBEDDING"
_MODULE = "sglang.srt.layers.vocab_parallel_embedding"
_patched = False


def _use_triton_embedding_off(self, input_) -> bool:
    return False


def _apply() -> None:
    try:
        import sglang.srt.layers.vocab_parallel_embedding as m

        cls = m.VocabParallelEmbedding
        if not hasattr(cls, "_use_triton_embedding"):
            _log.skipped("no _use_triton_embedding (SGLang < 0.5.18)")
            return
        cls._use_triton_embedding = _use_triton_embedding_off
        _log.applied("VocabParallelEmbedding: Triton embedding off -> torch mask+gather+fill")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch VocabParallelEmbedding: %s", exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    if os.getenv(_KEEP_KEY, "0").lower() in ("1", "true", "yes", "on"):
        _log.skipped("%s=1, keeping SGLang's Triton embedding kernel", _KEEP_KEY)
        return
    _install_post_import_wrap(_MODULE, _apply)
    _patched = True
