# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map ``torch.cuda`` onto TXDA (``torch_txda.transfer_to_txda``).

Importing ``torch_txda.transfer_to_txda`` runs its ``init()``: ``torch.cuda``
(streams, events, memory, ``is_available`` ...) and ``Tensor.cuda`` are
redirected to txda, so SGLang code that still talks to ``torch.cuda`` works on
TX8110. The plugin depends on this mapping everywhere, so it is an explicit,
logged step rather than a side effect of another module's import.

Runs at the same points as before: first in ``apply_all_txda_patches`` and
from ``platform_stubs.patch()`` (which the examples call at import time, before
SGLang is imported). Idempotent.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("cuda_alias")
_done = False


def patch() -> None:
    global _done
    if _done:
        return
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        _log.skipped("torch_txda not installed")
        return
    try:
        from torch_txda import transfer_to_txda  # noqa: F401  (import runs init())
    except Exception as exc:  # noqa: BLE001
        _log.failed("torch_txda.transfer_to_txda failed, torch.cuda is NOT mapped to txda: %s", exc)
        return
    _done = True
    _log.applied("torch.cuda -> txda (torch_txda.transfer_to_txda)")
