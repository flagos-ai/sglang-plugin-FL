"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace direct edits to sglang source that were previously required on
Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - attention_registry: defer CUDA-only linear-attn backend imports on NPU
  - qwen_vl_processor: transformers-version-compatible Qwen-VL preprocess
  - qwen36_causal_conv: single-sequence and equal-length batched prefill
  - qwen36_causal_conv_decode_fused: fuse BF16 decode conv and state update
  - vocab_parallel_embedding: fuse sharded lookup and zero masking
  - flagtree_driver: cache lazy driver utilities in affected FlagTree versions
"""

import logging

from .patches.attention_registry import patch_attn_backend_wrapper
from .patches.fia_tnd_gqa import patch_fia_tnd_gqa
from .patches.flagtree_driver import patch_flagtree_driver_utils
from .patches.prefill_tuning import patch_prefill_tuning
from .patches.qwen36_fla_solve import patch_qwen36_fla_solve
from .patches.qwen36_causal_conv import patch_qwen36_causal_conv_prefill
from .patches.qwen36_causal_conv_decode_fused import (
    patch_qwen36_causal_conv_decode_fused,
)
from .patches.qwen36_graph_sync import patch_qwen36_graph_sync
from .patches.qwen_vl_processor import patch_qwen_vl_processor
from .patches.vocab_parallel_embedding import patch_vocab_parallel_embedding
from .patches.scheduler_pp import (
    patch_pp_launch_batch_sync,
    patch_pp_send_recv_order,
)

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_ascend_patches() -> None:
    """Apply all Ascend/NPU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_flagtree_driver_utils()
    patch_pp_send_recv_order()
    patch_pp_launch_batch_sync()
    patch_attn_backend_wrapper()
    patch_qwen_vl_processor()
    patch_qwen36_causal_conv_prefill()
    patch_qwen36_causal_conv_decode_fused()
    patch_qwen36_graph_sync()
    patch_vocab_parallel_embedding()
    patch_fia_tnd_gqa()
    patch_prefill_tuning()
    patch_qwen36_fla_solve()


apply_ascend_patches()
