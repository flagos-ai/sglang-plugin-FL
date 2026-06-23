"""Vendor monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
Add one ``patch_xxx`` call per concern; put the implementation under ``patches/``.
"""

import logging

from .patches.supported_devices import patch_supported_devices

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_template_patches():
    """Apply all Template-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_supported_devices()


apply_template_patches()
