# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform device-support monkey-patches for sglang core.

Patches applied (all safe / idempotent — each is a no-op when not applicable):

  1. ``torch.library.Library._register_fake`` — wrap to suppress
     ``RuntimeError`` when a torchvision fake-implementation references
     an operator that does not exist on CPU-only torch builds (e.g. ``nms``).
     Works on all platforms.

  2. ``torch.cuda.get_device_properties`` — when CUDA is unavailable
     (CPU / txda), replace with a safe wrapper that returns a dummy
     properties object.  Prevents ``flashinfer`` import-time crashes.

  3. ``sglang.srt.utils.common.get_device()`` — wrap to return ``"txda"``
     when the original raises ``RuntimeError`` and txda hardware is detected.
     Works on all platforms — on non-txda the original error is re-raised.

  4. ``sglang.srt.utils.is_txda()`` — inject the txda device-detection
     utility into sglang.srt.utils if not already present.
     Only on txda.

  5. ``device_config.SUPPORTED_DEVICES`` — append ``"txda"``.
     Only on txda.
"""

import functools

import torch

from sglang_fl.dispatch.backends.vendor.txda.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.txda.patches._utils import is_txda as _is_txda

_log = patch_logger("device_support")

_patched = False
_originals: dict[str, object] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Internal patch helpers
# ──────────────────────────────────────────────────────────────────────────────


def _patch_torchvision_fake_register() -> None:
    """Wrap ``torch.library.Library._register_fake`` to tolerate missing ops.

    On CPU-only torch builds, torchvision attempts to register fake
    implementations for operators (e.g. ``nms``) that don't exist in the
    CPU build.  The original method raises ``RuntimeError("...does not exist")``
    which kills the import.  Wrapping it here silently returns ``None`` for
    that case.
    """
    try:
        orig = torch.library.Library._register_fake
    except AttributeError:
        _log.skipped("_register_fake not available on this torch version")
        return

    if getattr(orig, "_sglang_fl_patched", False):
        _log.skipped("_register_fake already patched")
        return

    @functools.wraps(orig)
    def _patched__register_fake(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except RuntimeError as e:
            if "does not exist" in str(e):
                return None
            raise

    _patched__register_fake._sglang_fl_patched = True  # type: ignore[attr-defined]
    torch.library.Library._register_fake = _patched__register_fake
    _originals["_register_fake"] = orig
    _log.applied("torch.library.Library._register_fake patched (missing-op tolerant)")


def _patch_safe_cuda_properties() -> None:
    """Replace ``torch.cuda.get_device_properties`` with a safe wrapper.

    When CUDA is unavailable the wrapper returns a dummy properties object
    so that ``flashinfer`` (which calls ``torch.cuda.get_device_properties(0)``
    at import time) can proceed without crashing.
    """
    if torch.cuda.is_available():
        _log.skipped("CUDA available — safe get_device_properties skipped")
        return

    orig = torch.cuda.get_device_properties
    if getattr(orig, "_sglang_fl_patched", False):
        _log.skipped("get_device_properties already patched")
        return

    @functools.wraps(orig)
    def _safe_get_device_properties(device=None):
        try:
            return orig(device or 0)
        except (AssertionError, RuntimeError):
            return type(
                "_DummyDeviceProps",
                (),
                {
                    "multi_processor_count": 1,
                    "name": "dummy",
                    "total_memory": 0,
                    "major": 0,
                    "minor": 0,
                },
            )()

    _safe_get_device_properties._sglang_fl_patched = True  # type: ignore[attr-defined]
    torch.cuda.get_device_properties = _safe_get_device_properties
    _originals["get_device_properties"] = orig
    _log.applied("torch.cuda.get_device_properties patched (safe fallback)")


def _patch_get_device_txda() -> None:
    """Wrap ``sglang.srt.utils.common.get_device()`` to recognise txda.

    The original raises ``RuntimeError`` when no known device is found.
    This wrapper catches that and returns ``"txda"`` when txda hardware
    is available.
    """
    try:
        import sglang.srt.utils.common as _sglang_common
    except Exception as exc:
        _log.failed("failed to import sglang.srt.utils.common: %s", exc)
        return

    if hasattr(_sglang_common, "_sglang_fl_get_device_patched"):
        _log.skipped("get_device already patched")
        return

    orig_get_device = _sglang_common.get_device

    @functools.wraps(orig_get_device)
    def _patched_get_device():
        try:
            return orig_get_device()
        except RuntimeError:
            if _is_txda():
                return "txda"
            raise

    _sglang_common.get_device = _patched_get_device
    _sglang_common._sglang_fl_get_device_patched = True  # type: ignore[attr-defined]
    _originals["get_device"] = orig_get_device
    _log.applied("sglang.srt.utils.common.get_device patched → txda aware")


# ──────────────────────────────────────────────────────────────────────────────
# TXDA-specific helpers (only applied when txda is available)
# ──────────────────────────────────────────────────────────────────────────────


def _patch_is_txda_util() -> None:
    """Inject ``is_txda()`` into ``sglang.srt.utils`` if not already present."""
    try:
        import sglang.srt.utils as _utils

        if not hasattr(_utils, "is_txda"):

            @functools.lru_cache(maxsize=1)
            def _is_txda_func() -> bool:
                try:
                    import torch_txda  # noqa: F401
                except ImportError:
                    return False
                return hasattr(torch, "txda") and torch.txda.is_available()

            _utils.is_txda = _is_txda_func
            _log.applied("injected is_txda() into sglang.srt.utils")
        else:
            _log.skipped("is_txda() already present in sglang.srt.utils")
    except Exception as exc:
        _log.failed("failed to inject is_txda(): %s", exc)


def _patch_supported_devices() -> None:
    """Append ``"txda"`` to ``device_config.SUPPORTED_DEVICES``."""
    try:
        from sglang.srt.configs import device_config

        _originals.setdefault("supported_devices", list(device_config.SUPPORTED_DEVICES))

        if "txda" not in device_config.SUPPORTED_DEVICES:
            device_config.SUPPORTED_DEVICES.append("txda")
            _log.applied(
                "added 'txda' to SUPPORTED_DEVICES (%s)",
                device_config.SUPPORTED_DEVICES,
            )
        else:
            _log.skipped("'txda' already in SUPPORTED_DEVICES")
    except Exception as exc:
        _log.failed("failed to patch SUPPORTED_DEVICES: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def patch() -> None:
    """Apply all device-support monkey-patches.

    Patches 1–3 run on every platform (safe no-ops when not applicable).
    Patches 4–5 only run when txda hardware is detected.

    Idempotent: safe to call multiple times.
    """
    global _patched
    if _patched:
        return

    # Universal patches (always safe)
    _patch_torchvision_fake_register()
    _patch_safe_cuda_properties()
    _patch_get_device_txda()

    # TXDA-specific extensions
    if _is_txda():
        _patch_is_txda_util()
        _patch_supported_devices()

    _patched = True


def restore() -> None:
    """Restore original device_config.SUPPORTED_DEVICES (best-effort)."""
    global _patched
    if not _patched:
        return
    try:
        from sglang.srt.configs import device_config

        orig = _originals.get("supported_devices")
        if orig is not None:
            device_config.SUPPORTED_DEVICES[:] = orig
        _patched = False
        _log.applied("restored original SUPPORTED_DEVICES")
    except Exception as exc:
        _log.failed("failed to restore SUPPORTED_DEVICES: %s", exc)
