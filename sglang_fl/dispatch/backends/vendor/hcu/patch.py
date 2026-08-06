"""HCU monkey-patches on SGLang internals."""

from .patches.cuda_graph_pre_sample_fence import setup_cuda_graph_pre_sample_fence
from .patches.fla_dispatch_filter import patch_fla_dispatch_filter

_patches_applied = False


def apply_hcu_patches() -> None:
    """Apply all HCU-specific patches once."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_fla_dispatch_filter()
    setup_cuda_graph_pre_sample_fence()


apply_hcu_patches()
