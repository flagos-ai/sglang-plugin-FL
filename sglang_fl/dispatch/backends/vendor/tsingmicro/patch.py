# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA vendor monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl._apply_vendor_patches()`` when TXDA hardware is
detected.  Add one ``patch_xxx`` call per concern; put the implementation
under ``patches/``.

Patches are idempotent — safe to import multiple times.
"""

import logging

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import apply_all_txda_patches

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_txda_patches():
    """Apply all TXDA-specific monkey-patches.  Idempotent."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    apply_all_txda_patches()


# Auto-execute on import (the standard vendor-patch convention).
apply_txda_patches()
