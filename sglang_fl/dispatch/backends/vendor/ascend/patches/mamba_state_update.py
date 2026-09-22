"""CANN 8.5 compatibility for speculative Mamba state updates.

The official 2026.8.10 NPU kernel release launches
``move_cache_dynamic_last_kernel_h_block`` with the Triton Ascend compiler's
default auto multi-buffering.  Qwen3.6-27B MTP uses a 2 x 128 x 128 tile;
duplicating that tile exceeds the 910C unified-buffer budget.  Disable
multi-buffering only for this kernel and leave every other NPU kernel's
compiler policy unchanged.
"""

from __future__ import annotations

from functools import wraps
import importlib
import logging
from typing import Any


logger = logging.getLogger(__name__)

_KERNEL_MODULE = "sgl_kernel_npu.mamba.mamba_state_update_triton"
_KERNEL_NAME = "move_cache_dynamic_last_kernel_h_block"
_PATCH_MARKER = "_sglang_fl_cann85_multibuffer_disabled"


def patch_mamba_state_update_multibuffer() -> bool:
    """Force ``multibuffer=False`` for the one UB-overflowing MTP kernel."""

    try:
        module = importlib.import_module(_KERNEL_MODULE)
        kernel = getattr(module, _KERNEL_NAME)
        original_run = kernel.run
    except (AttributeError, ImportError):
        logger.debug("NPU Mamba state-update kernel is unavailable", exc_info=True)
        return False

    if getattr(original_run, _PATCH_MARKER, False):
        return True

    @wraps(original_run)
    def run_without_multibuffer(*args: Any, **kwargs: Any) -> Any:
        kwargs["multibuffer"] = False
        return original_run(*args, **kwargs)

    setattr(run_without_multibuffer, _PATCH_MARKER, True)
    kernel.run = run_without_multibuffer
    logger.info(
        "Disabled Triton auto multi-buffering for %s.%s",
        _KERNEL_MODULE,
        _KERNEL_NAME,
    )
    return True

