"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace the compatibility edits still required on Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - vision: use SGLang's SDPA fallback when CANN 8.5 fused attention cannot
    accept an unaligned head dimension
  - mamba_state_update: disable auto multi-buffering for the Qwen3.6 MTP
    state-copy tile that exceeds the 910C unified-buffer budget
  - logsumexp: disable auto multi-buffering for SGLang's fused logprob top-k
    kernel, which otherwise crashes the CANN 8.5 BiSheng compiler on 910C

SGLang v0.5.18 already contains the current NPU attention-wrapper and Qwen-VL
processor implementations. Keeping the old plugin replacements would discard
new v0.5.18 behavior, so those replacements are intentionally not applied.
"""

import logging

from .patches.logsumexp import patch_logsumexp_topk_multibuffer
from .patches.scheduler_pp import (
    patch_pp_launch_batch_sync,
    patch_pp_send_recv_order,
)
from .patches.mamba_state_update import patch_mamba_state_update_multibuffer
from .patches.vision import patch_vision_ascend_attention

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_ascend_patches() -> None:
    """Apply all Ascend/NPU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    patch_pp_send_recv_order()
    patch_pp_launch_batch_sync()
    patch_vision_ascend_attention()
    patch_mamba_state_update_multibuffer()
    patch_logsumexp_topk_multibuffer()
    _patches_applied = True


apply_ascend_patches()
