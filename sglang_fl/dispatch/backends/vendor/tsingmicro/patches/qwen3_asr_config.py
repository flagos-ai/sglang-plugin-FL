# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch ``AutoConfig.register`` to default ``exist_ok=True``.

Transformers 5.15+ ships built-in ``qwen3_asr`` / ``qwen3_asr_thinker`` model
configs.  When sglang subsequently calls ``AutoConfig.register("qwen3_asr", ...)``
(without ``exist_ok=True``), the duplicate registration raises ``ValueError``.

This patch wraps ``AutoConfig.register`` so that ``exist_ok`` defaults to
``True``, making the duplicate a silent no-op while preserving all other
validation (model_type consistency, etc.).

Applied unconditionally — safe on all platforms / transformers versions.
"""

import functools

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("qwen3_asr_config")

_patched = False
_originals: dict[str, object] = {}


def patch() -> None:
    """Patch ``AutoConfig.register`` to tolerate duplicate registrations."""
    global _patched
    if _patched:
        return

    try:
        from transformers.models.auto.configuration_auto import AutoConfig
    except Exception as exc:
        _log.failed("failed to import AutoConfig: %s", exc)
        return

    orig_register = AutoConfig.register
    if getattr(orig_register, "_sglang_fl_patched", False):
        _log.skipped("AutoConfig.register already patched")
        _patched = True
        return

    @functools.wraps(orig_register)
    def _patched_register(model_type, config, exist_ok=True):
        """Wrapper that defaults ``exist_ok=True`` to tolerate duplicate model registrations.

        Transformers 5.15+ registers ``qwen3_asr`` internally; sglang's own
        ``qwen3_asr.py`` also registers it, causing a ``ValueError`` without
        this patch.
        """
        return orig_register(model_type, config, exist_ok=exist_ok)

    _patched_register._sglang_fl_patched = True  # type: ignore[attr-defined]
    AutoConfig.register = staticmethod(_patched_register)
    _originals["register"] = orig_register
    _log.applied("AutoConfig.register patched → exist_ok=True by default")
    _patched = True


def restore() -> None:
    """Restore original ``AutoConfig.register`` (best-effort)."""
    global _patched
    if not _patched:
        return
    try:
        from transformers.models.auto.configuration_auto import AutoConfig

        orig = _originals.get("register")
        if orig is not None:
            AutoConfig.register = orig
        _patched = False
        _log.applied("restored original AutoConfig.register")
    except Exception as exc:
        _log.failed("failed to restore AutoConfig.register: %s", exc)
