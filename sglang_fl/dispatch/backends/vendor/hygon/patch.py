"""HCU monkey-patches on SGLang internals."""

from .register_platform import register_hcu_attention_backend
from .patches.pp_output_launch_dependency import setup_pp_output_launch_dependency
from .patches.fla_dispatch_filter import patch_fla_dispatch_filter
from .lifecycle import apply_hcu_lifecycle_patches
from .patches.speculative_backend import register_hcu_speculative_backends
from .patches.mtp_weights import register_hcu_mtp_weight_guard
from .patches.mtp_logprobs import register_hcu_mtp_logprob_guard

_patches_applied = False


def apply_hcu_patches() -> None:
    """Apply all HCU-specific patches once."""
    global _patches_applied
    if _patches_applied:
        return
    # launch_server loads general plugins before parsing its CLI. Register here
    # rather than waiting for PlatformFL.init_backend in the model worker.
    register_hcu_attention_backend()
    register_hcu_mtp_weight_guard()
    register_hcu_mtp_logprob_guard()
    register_hcu_speculative_backends()
    patch_fla_dispatch_filter()
    setup_pp_output_launch_dependency()
    apply_hcu_lifecycle_patches()
    _patches_applied = True


apply_hcu_patches()
