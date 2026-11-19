# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Install the Qwen3.8 MUSA model hooks once through SGLang's hook registry."""

import logging
from importlib.util import find_spec

logger = logging.getLogger(__name__)
_registered = False


def register_hooks():
    global _registered
    if _registered:
        return
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    if info is None or info.vendor_name != "mthreads":
        return
    if find_spec("sglang.srt.models.qwen4_exp") is None:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry
    from . import (
        gdn_norm,
        qwen4_communication,
        qwen4_hc,
        qwen4_qsa,
        qwen4_qsa_rope,
    )

    entries = (
        qwen4_hc.hooks()
        + qwen4_qsa.hooks()
        + qwen4_qsa_rope.hooks()
        + qwen4_communication.hooks()
    )
    from sglang.kernels.fused_op import BaseFusedOp

    # HookRegistry defers application. Preserve unrelated pending hooks if this
    # model's registration group fails before application.
    previous = {
        target: list(HookRegistry._hooks.get(target, ())) for target, _, _ in entries
    }
    previous_oot = BaseFusedOp._oot_forward_registry.get("oot")
    if previous_oot is not None:
        previous_oot = dict(previous_oot)
    try:
        gdn_norm.register_hooks()
        for target, hook, kind in entries:
            HookRegistry.register(target, hook, kind)
    except Exception:
        for target, hooks in previous.items():
            if hooks:
                HookRegistry._hooks[target] = hooks
            else:
                HookRegistry._hooks.pop(target, None)
        if previous_oot is None:
            BaseFusedOp._oot_forward_registry.pop("oot", None)
        else:
            BaseFusedOp._oot_forward_registry["oot"] = previous_oot
        raise
    _registered = True
    logger.info("Qwen3.8 MUSA HC and QSA hooks registered")
