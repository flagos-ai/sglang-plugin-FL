# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform device-support monkey-patch for sglang core (SGLang 0.5.18).

Patch applied:

  ``torch.library.Library._register_fake`` — wrap to suppress
  ``RuntimeError`` when a torchvision fake-implementation references an
  operator that does not exist on CPU-only torch builds (e.g. ``nms``; the
  TXDA torch is a +cpu build). Idempotent.

Removed for 0.5.18 (dead or covered upstream): a safe
``torch.cuda.get_device_properties`` (never applied: torch.cuda is mapped to
txda, so CUDA reports available), a ``get_device()`` wrapper (upstream returns
without raising on txda, and ``server_args`` imports the function by name), an
injected ``sglang.srt.utils.is_txda`` (no reader), and appending ``"txda"`` to
``device_config.SUPPORTED_DEVICES`` (out-of-tree platforms are accepted as is).
"""

import functools

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

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


def patch() -> None:
    """Apply the device-support patch. Idempotent."""
    global _patched
    if _patched:
        return
    _patch_torchvision_fake_register()
    _patched = True
