"""GCU monkey-patches on SGLang internals (enflame/tops1.10.6, sglang 0.5.18).

Vendor subpatches are loaded individually: the exp/0.5.18 tree drifted from the
0.5.18 release wheel for three modules that target sglang dev-only APIs
(vision -> sglang.srt.layers.dp_attention.get_attention_tp_size,
cuda_graph_runner -> sglang.srt.model_executor.breakable_cuda_graph,
chunk_delta_h -> sglang.srt.layers.attention.fla) — absent from the 0.5.18
release wheel. Guarding each import means one stale subpatch cannot disable the
whole vendor layer (as it did: the layer never applied, and the scheduler died
on the is_integrated read instead). The device-properties compat fix loads
first; the rest apply when importable.
"""

import importlib
import logging

from .patches.device_properties import patch_gcu_device_properties

logger = logging.getLogger(__name__)
_patches_applied = False

_PKG = "sglang_fl.dispatch.backends.vendor.enflame"


def apply_gcu_patches():
    """Apply all GCU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Expose is_integrated=False on gcu device props (triton_load_watch /
    # get_available_gpu_memory serve blocker).
    patch_gcu_device_properties()

    def _apply(modname, fnname):
        try:
            mod = importlib.import_module(f"{_PKG}.patches.{modname}")
        except ImportError as e:
            logger.warning("gcu vendor patch skipped (%s): %s", modname, e)
            return
        try:
            getattr(mod, fnname)()
            logger.info("gcu vendor patch applied: %s", modname)
        except Exception as e:  # noqa: BLE001
            logger.warning("gcu vendor patch apply failed (%s): %r", modname, e)

    _apply("scheduler_pp_mixin", "patch_scheduler_pp_mixin")
    _apply("supported_devices", "patch_supported_devices")
    _apply("device_communicators", "patch_communicator_hooks")
    _apply("flashattention_backend", "patch_flashattention_backend")
    # patches.vision: stale for 0.5.18 (get_attention_tp_size) — skipped via
    # ImportError guard above.
    _apply("parallel_state", "patch_parallel_state")
    _apply("qwen3_vl", "patch_qwen3_vl")
    # patches.cuda_graph_runner: stale for 0.5.18 (breakable_cuda_graph).
    _apply("mem_cache", "patch_mem_cache")
    _apply("causal_conv1d", "patch_causal_conv1d_fn")
    # patches.chunk_delta_h: stale for 0.5.18 (layers.attention.fla).


apply_gcu_patches()
