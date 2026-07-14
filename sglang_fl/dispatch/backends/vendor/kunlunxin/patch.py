"""Vendor monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
Add one ``patch_xxx`` call per concern; put the implementation under ``patches/``.
"""

import logging

from .patches.causal_conv1d import patch_causal_conv1d
from .patches.clamp_position import patch_clamp_position
from .patches.pp_send_first import patch_pp_send_recv_and_preprocess_output_tensors
from .patches.suppress_pynccl import patch_suppress_pynccl

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_kunlunxin_patches():
    """Apply all kunlunxin-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_clamp_position()
    patch_causal_conv1d()
    patch_suppress_pynccl()
    patch_pp_send_recv_and_preprocess_output_tensors()


apply_kunlunxin_patches()
