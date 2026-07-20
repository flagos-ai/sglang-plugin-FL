import logging

from .patches.supported_devices import patch_supported_devices
from .patches.device_communicators import patch_communicator_hooks
from .patches.flashattention_backend import patch_flashattention_backend
from .patches.vision import patch_vision_flashattention_backend
from .patches.server_args import patch_server_args
from .patches.parallel_state import patch_parallel_state
from .patches.qwen3_vl import patch_qwen3_vl

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_gcu_patches():
    """Apply all Template-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_server_args()
    patch_supported_devices()
    patch_communicator_hooks()
    patch_flashattention_backend()
    patch_vision_flashattention_backend()
    patch_parallel_state()
    patch_qwen3_vl()

apply_gcu_patches()
