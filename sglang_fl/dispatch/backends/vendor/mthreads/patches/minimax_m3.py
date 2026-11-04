# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads compatibility fixes for MiniMax-M3 on SGLang 0.5.18.

The v0.5.18 KVCacheConfigurator selects the OOT path before the native MiniMax
pool. No public hook in commit 71de97b264b04dcd514cf904003028aefe9775c8
expresses model-specific OOT allocation. Remove this private-method wrapper
when upstream offers that hook or selects the MiniMax pool before OOT.
Numeric kernels remain in SGLang, FlagGems, or the vendor runtime.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_sglang_fl_musa_minimax_m3"


def _validate_signature(callable_obj, required_parameters: tuple[str, ...]) -> None:
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    missing = [name for name in required_parameters if name not in parameters]
    if missing:
        raise RuntimeError(
            f"MiniMax-M3 patch target {callable_obj!r} is missing parameters: "
            + ", ".join(missing)
        )
    if tuple(parameters) != required_parameters:
        raise RuntimeError(
            f"MiniMax-M3 patch target {callable_obj!r} has an unsupported signature"
        )
    try:
        signature.bind(object(), max_total_num_tokens=0)
    except TypeError as exc:
        raise RuntimeError(
            f"MiniMax-M3 patch target {callable_obj!r} has an unsupported signature"
        ) from exc


def _build_kv_pool_wrapper(original, kv_module):
    @wraps(original)
    def build_oot_mha_kv_pool(self, *, max_total_num_tokens: int):
        hf_config = self.model_config.hf_config
        if str(self.device).startswith("musa") and kv_module.is_minimax_sparse(
            hf_config
        ):
            native = getattr(type(self), "_build_minimax_sparse_kv_pool", None)
            if not callable(native):
                raise RuntimeError(
                    "MUSA MiniMax requires the native sparse KV pool API"
                )
            _validate_signature(native, ("self", "max_total_num_tokens"))
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
    if original_kv is None:
        logger.info("MiniMax-M3 support is unavailable in this SGLang build")
        return False

    if getattr(original_kv, _PATCH_MARKER, False):
        return False

    _validate_signature(original_kv, ("self", "max_total_num_tokens"))
    if native_minimax_kv is not None:
        _validate_signature(native_minimax_kv, ("self", "max_total_num_tokens"))

    kv_cls._build_oot_mha_kv_pool = _build_kv_pool_wrapper(original_kv, kv_module)

    logger.info("MUSA MiniMax-M3 KV-pool routing patch applied")
    return True
