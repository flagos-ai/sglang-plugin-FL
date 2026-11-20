import logging

from .patches.supported_devices import patch_supported_devices
from .patches.device_communicators import patch_communicator_hooks
from .patches.flashattention_backend import patch_flashattention_backend
from .patches.vision import patch_vision_flashattention_backend
from .patches.parallel_state import patch_parallel_state
from .patches.qwen3_vl import patch_qwen3_vl
from .patches.mem_cache import patch_mem_cache
from .patches.causal_conv1d import patch_causal_conv1d_fn
from .patches.chunk_delta_h import patch_chunk_delta_h
from .patches.scheduler_pp_mixin import patch_scheduler_pp_mixin
from .patches.runtime_memory import patch_runtime_memory
from .patches.buffer_copy import patch_buffer_copy
from .patches.flagcx_allreduce import patch_flagcx_allreduce
from .patches.multimodal_mask import patch_multimodal_mask
from .patches.mem_cache.mtp_state import patch_mtp_state
from .patches.speculative import patch_speculative
from .patches.logprob import patch_logprob
from .patches.mtp_graph import patch_mtp_graph
from .patches.topk import patch_topk

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_gcu_patches():
    """Apply all Template-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_scheduler_pp_mixin()
    patch_supported_devices()
    patch_runtime_memory()
    patch_communicator_hooks()
    patch_flagcx_allreduce()
    patch_flashattention_backend()
    patch_vision_flashattention_backend()
    patch_parallel_state()
    patch_qwen3_vl()
    patch_mem_cache()
    patch_buffer_copy()
    patch_multimodal_mask()
    patch_causal_conv1d_fn()
    patch_chunk_delta_h()
    patch_mtp_state()
    patch_speculative()
    patch_logprob()
    patch_mtp_graph()
    patch_topk()

apply_gcu_patches()
