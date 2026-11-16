"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace the compatibility edits still required on Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - vision: use SGLang's SDPA fallback when CANN 8.5 fused attention cannot
    accept an unaligned head dimension
  - vision_graph: preserve eager position interpolation and replay output ownership
  - linear: expose configured mm dispatch for unbiased NPU BF16 matrix inputs
  - qwen35_pp_rotary: refresh position caches on each pipeline stage
  - mamba_state_update: disable auto multi-buffering for the Qwen3.6 MTP
    state-copy tile that exceeds the 910C unified-buffer budget
  - gdn_target_verify: make top-k=1 MTP verification share ordinary decode's
    per-token numerical persistence boundaries
  - logsumexp: split logprob top-k into SGLang's ordinary row normalizer and
    PyTorch top-k because CANN 8.5 cannot compile the fused kernel on 910C
  - req_to_token_pool: use the PyTorch writer on FlagTree 0.7.0 / CANN 9.2.1
  - gdn_prefill: adapt gating and state layouts to the sgl-kernel-npu 2026.9.0 ABI

SGLang v0.5.18 already contains the current NPU attention-wrapper and Qwen-VL
processor implementations. Keeping the old plugin replacements would discard
new v0.5.18 behavior, so those replacements are intentionally not applied.
"""

import logging

from .patches.logsumexp import patch_logsumexp_topk_fallback
from .patches.linear import patch_unquantized_linear_mm
from .patches.gdn_prefill import patch_gdn_layout
from .patches.gdn_target_verify import patch_gdn_target_verify
from .patches.scheduler_pp import (
    patch_pp_launch_batch_sync,
    patch_pp_send_recv_order,
)
from .patches.mamba_state_update import patch_mamba_state_update_multibuffer
from .patches.vision import patch_vision_ascend_attention
from .patches.vision_graph import patch_qwen3_vl_npu_graph
from .patches.qwen35_pp_rotary import patch_qwen35_pp_rotary_cache
from .patches.req_to_token_pool import patch_request_pool_write_fallback

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
    patch_qwen3_vl_npu_graph()
    patch_unquantized_linear_mm()
    patch_qwen35_pp_rotary_cache()
    patch_mamba_state_update_multibuffer()
    patch_gdn_layout()
    patch_gdn_target_verify()
    patch_logsumexp_topk_fallback()
    patch_request_pool_write_fallback()
    _patches_applied = True


apply_ascend_patches()
