"""HCU monkey-patches on SGLang internals."""

from .patches.fla_dispatch_filter import patch_fla_dispatch_filter

_patches_applied = False


def apply_hcu_patches() -> None:
    """Apply all HCU-specific patches once."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_fla_dispatch_filter()


apply_hcu_patches()
