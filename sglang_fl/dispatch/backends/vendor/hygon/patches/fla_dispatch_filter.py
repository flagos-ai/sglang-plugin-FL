"""Restore OOT-filtered FLA functions after the common bridge patch."""

from __future__ import annotations

import importlib
import logging
import os

from sglang_fl.dispatch.config import get_effective_config
from sglang_fl.dispatch.fla_patch import get_original

logger = logging.getLogger(__name__)

_FLA_TARGETS = {
    "chunk_gated_delta_rule": (
        "sglang.srt.layers.attention.fla.chunk",
        "chunk_gated_delta_rule",
    ),
    "fused_recurrent_gated_delta_rule": (
        "sglang.srt.layers.attention.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule",
    ),
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "sglang.srt.layers.attention.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
}

_GDN_TARGETS = {
    "chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "fused_recurrent_gated_delta_rule_packed_decode"
    ),
}


def _parse_list(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _get_oot_filters() -> tuple[set[str], set[str]]:
    config = get_effective_config()

    whitelist = _parse_list(os.environ.get("SGLANG_FL_OOT_WHITELIST", ""))
    blacklist_env = os.environ.get("SGLANG_FL_OOT_BLACKLIST", "").strip()
    if blacklist_env:
        blacklist = _parse_list(blacklist_env)
    else:
        blacklist = set(config.get("oot_blacklist", []) or [])

    if whitelist and blacklist:
        raise ValueError(
            "Cannot set both SGLANG_FL_OOT_WHITELIST and "
            "SGLANG_FL_OOT_BLACKLIST."
        )
    return whitelist, blacklist


def _is_filtered(
    name: str,
    whitelist: set[str],
    blacklist: set[str],
) -> bool:
    if whitelist and name not in whitelist:
        return True
    return name in blacklist


def patch_fla_dispatch_filter() -> tuple[str, ...]:
    """Restore original FLA bindings excluded from HCU OOT dispatch."""
    whitelist, blacklist = _get_oot_filters()
    filtered = {
        name
        for name in _FLA_TARGETS
        if _is_filtered(name, whitelist, blacklist)
    }
    if not filtered:
        return ()

    originals = {name: get_original(name) for name in filtered}
    available = {name: fn for name, fn in originals.items() if fn is not None}
    if not available:
        logger.info(
            "Skipped HCU FLA filter patch because the common FLA patch is inactive"
        )
        return ()

    restored = []
    for name, original in available.items():
        module_name, attribute = _FLA_TARGETS[name]
        module = importlib.import_module(module_name)
        setattr(module, attribute, original)
        restored.append(name)

    gdn_names = set(available).intersection(_GDN_TARGETS)
    if gdn_names:
        try:
            gdn_triton = importlib.import_module(
                "sglang.srt.layers.attention.linear.kernels.gdn_triton"
            )
            for name in gdn_names:
                setattr(gdn_triton, _GDN_TARGETS[name], available[name])
        except Exception as exc:
            logger.warning("Failed to restore HCU gdn_triton FLA bindings: %s", exc)

    restored.sort()
    logger.info("Restored OOT-filtered HCU FLA functions: %s", restored)
    return tuple(restored)
