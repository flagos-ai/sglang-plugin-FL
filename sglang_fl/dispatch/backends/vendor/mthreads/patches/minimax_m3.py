# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads compatibility fixes for MiniMax-M3 on SGLang 0.5.18.

The patches in this module only select framework-owned behavior. Numeric
kernels remain in SGLang, FlagGems, or the vendor runtime.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_sglang_fl_musa_minimax_m3"


def _validate_signature(callable_obj, required_parameters: tuple[str, ...]) -> None:
    parameters = inspect.signature(callable_obj).parameters
    missing = [name for name in required_parameters if name not in parameters]
    if missing:
        raise RuntimeError(
            f"MiniMax-M3 patch target {callable_obj!r} is missing parameters: "
            + ", ".join(missing)
        )


def _build_kv_pool_wrapper(original, kv_module):
    @wraps(original)
    def build_oot_mha_kv_pool(self, *, max_total_num_tokens: int):
        hf_config = self.model_config.hf_config
        if str(self.device).startswith("musa") and kv_module.is_minimax_sparse(
            hf_config
        ):
            return self._build_minimax_sparse_kv_pool(
                max_total_num_tokens=max_total_num_tokens
            )
        return original(self, max_total_num_tokens=max_total_num_tokens)

    setattr(build_oot_mha_kv_pool, _PATCH_MARKER, True)
    return build_oot_mha_kv_pool


def install_minimax_m3_patches(kv_module=None) -> bool:
    """Install the model-scoped MUSA KV-pool routing patch idempotently."""

    if kv_module is None:
        try:
            from sglang.srt.mem_cache import kv_cache_configurator as imported_kv
        except ImportError:
            logger.info("MiniMax-M3 support is unavailable in this SGLang build")
            return False
        kv_module = imported_kv

    kv_cls = getattr(kv_module, "KVCacheConfigurator", None)
    required_capabilities = (
        kv_cls,
        getattr(kv_module, "is_minimax_sparse", None),
    )
    if any(capability is None for capability in required_capabilities):
        logger.info("MiniMax-M3 support is unavailable in this SGLang build")
        return False

    original_kv = getattr(kv_cls, "_build_oot_mha_kv_pool", None)
    native_minimax_kv = getattr(kv_cls, "_build_minimax_sparse_kv_pool", None)
    if any(target is None for target in (original_kv, native_minimax_kv)):
        logger.info("MiniMax-M3 support is unavailable in this SGLang build")
        return False

    if getattr(original_kv, _PATCH_MARKER, False):
        return False

    _validate_signature(original_kv, ("self", "max_total_num_tokens"))
    _validate_signature(native_minimax_kv, ("self", "max_total_num_tokens"))

    kv_cls._build_oot_mha_kv_pool = _build_kv_pool_wrapper(original_kv, kv_module)

    logger.info("MUSA MiniMax-M3 KV-pool routing patch applied")
    return True
