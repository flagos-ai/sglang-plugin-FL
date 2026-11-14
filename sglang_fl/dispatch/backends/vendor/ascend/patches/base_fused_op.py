"""Ascend registration for fused operators that no longer use MultiPlatformOp."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_registered = False


def patch_unquantized_fused_moe() -> None:
    """Register the native NPU MoE forward through BaseFusedOp.

    Newer SGLang versions derive ``UnquantizedFusedMoEMethod`` directly from
    ``BaseFusedOp``. The plugin's ``MultiPlatformOp`` AROUND hook therefore no
    longer observes this call site. Keep this registration in the Ascend patch
    module so other plugin platforms are unaffected.
    """

    global _registered
    if _registered:
        return

    try:
        from sglang.kernels.fused_op import BaseFusedOp
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
        )
    except (ImportError, AttributeError) as exc:
        logger.info("BaseFusedOp NPU MoE registration is unavailable: %s", exc)
        return

    register = getattr(BaseFusedOp, "register_oot_forward", None)
    forward_npu = getattr(UnquantizedFusedMoEMethod, "forward_npu", None)
    if not callable(register) or not callable(forward_npu):
        logger.info("BaseFusedOp NPU MoE registration API is unavailable")
        return

    register(UnquantizedFusedMoEMethod, forward_npu, "oot")
    _registered = True
    logger.info("Registered BaseFusedOp NPU forward for UnquantizedFusedMoEMethod")
