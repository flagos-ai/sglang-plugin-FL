# Copyright (c) 2026 BAAI. All rights reserved.
"""Route SGLang FLA entry points through the existing FlagOS dispatch bridge."""

import importlib
import logging

logger = logging.getLogger(__name__)

_originals = {}
_patched = False


def get_original(name: str):
    """Return the unpatched SGLang implementation for a backend fallback."""
    return _originals.get(name)


def _fla_modules():
    candidates = (
        "sglang.kernels.ops.attention.fla",
        "sglang.srt.layers.attention.fla",
    )
    for prefix in candidates:
        try:
            return (
                importlib.import_module(f"{prefix}.chunk"),
                importlib.import_module(f"{prefix}.fused_recurrent"),
            )
        except ModuleNotFoundError as exc:
            # A missing optional FLA package permits the older layout. A
            # missing dependency inside an existing package must be raised.
            if not (
                prefix.startswith(exc.name + ".")
                or exc.name in {prefix, f"{prefix}.chunk", f"{prefix}.fused_recurrent"}
            ):
                raise
    return None


def patch_fla_functions():
    """Install FLA bridges once, rolling back every changed entry on error."""
    global _patched
    if _patched:
        return _originals

    modules = _fla_modules()
    if modules is None:
        logger.info("SGLang FLA modules absent; FLA dispatch patch skipped")
        return None
    chunk_module, recurrent_module = modules
    from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge
    from sglang_fl.dispatch.bridge.fla_fused_recurrent import (
        fused_recurrent_gated_delta_rule_bridge,
    )
    from sglang_fl.dispatch.bridge.fla_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode_bridge,
    )

    targets = [
        (chunk_module, "chunk_gated_delta_rule", chunk_gated_delta_rule_bridge),
        (
            recurrent_module,
            "fused_recurrent_gated_delta_rule",
            fused_recurrent_gated_delta_rule_bridge,
        ),
        (
            recurrent_module,
            "fused_recurrent_gated_delta_rule_packed_decode",
            fused_recurrent_gated_delta_rule_packed_decode_bridge,
        ),
    ]
    gdn_path = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
    try:
        gdn_module = importlib.import_module(gdn_path)
    except ModuleNotFoundError as exc:
        if exc.name != gdn_path and not gdn_path.startswith(exc.name + "."):
            raise
    else:
        targets.extend(
            [
                (gdn_module, "chunk_gated_delta_rule", chunk_gated_delta_rule_bridge),
                (
                    gdn_module,
                    "fused_recurrent_gated_delta_rule_packed_decode",
                    fused_recurrent_gated_delta_rule_packed_decode_bridge,
                ),
            ]
        )

    saved = [(module, name, getattr(module, name)) for module, name, _ in targets]
    try:
        for module, name, bridge in targets:
            setattr(module, name, bridge)
    except Exception:
        for module, name, original in saved:
            setattr(module, name, original)
        raise

    _originals.update(
        {
            "chunk_gated_delta_rule": saved[0][2],
            "fused_recurrent_gated_delta_rule": saved[1][2],
            "fused_recurrent_gated_delta_rule_packed_decode": saved[2][2],
        }
    )
    _patched = True
    logger.info("SGLang FLA dispatch bridges installed")
    return _originals
