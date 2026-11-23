"""Keep the hybrid-mamba radix cache on no_buffer for MetaX.

SGLang >= 0.5.18 resolves --mamba-radix-cache-strategy auto to
extra_buffer when overlap scheduling is on, and extra_buffer asserts
is_cuda() (its FLA kernels are gated on the CUDA flag). MetaX reports
is_cuda() == False, so Qwen3.6 fails at startup unless the radix cache is
disabled. Reporting extra_buffer as unsupported makes auto resolve to
no_buffer with overlap scheduling off, which is what SGLang 0.5.11 used.
An explicit --mamba-radix-cache-strategy extra_buffer still fails loudly.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.arg_groups.overrides"
_patched = False


def patch_mamba_extra_buffer() -> None:
    global _patched
    if _patched:
        return
    try:
        module = importlib.import_module(_MODULE)
    except ImportError:
        return  # SGLang <= 0.5.15 has no extra_buffer auto selection
    if not hasattr(module, "supports_mamba_cache_extra_buffer"):
        return

    def supports_mamba_cache_extra_buffer_metax(view, model_arch: str) -> bool:
        return False

    module.supports_mamba_cache_extra_buffer = supports_mamba_cache_extra_buffer_metax
    _patched = True
    logger.info(
        "patched mamba radix cache: extra_buffer unsupported on MetaX (auto -> no_buffer)"
    )
