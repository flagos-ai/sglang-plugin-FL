# Copyright (c) 2026 BAAI. All rights reserved.
"""Monkey-patch SGLang FLA functions to use dispatch mechanism."""

import logging

logger = logging.getLogger(__name__)

# Store original (unpatched) FLA functions so backends can call them without recursion.
_originals = {}


def get_original(name: str):
    """Get the original (unpatched) FLA function by name."""
    return _originals.get(name)


def _select_fla_implementations(
    originals, bridges, *, vendor_only: bool, is_vendor_available=None
):
    """Choose bridge or original independently for each FLA function."""

    if not vendor_only:
        return dict(bridges)
    if not callable(is_vendor_available):
        raise ValueError("vendor_only selection requires an availability callback")
    return {
        name: bridges[name] if is_vendor_available(name) else originals[name]
        for name in bridges
    }


def patch_fla_functions(*, vendor_only: bool = False):
    """
    Replace SGLang's FLA module-level functions with dispatch bridges.

    In platform profiling, only functions with an available vendor adapter are
    replaced; all others remain on their original SGLang implementation.
    """
    try:
        # Import SGLang FLA modules
        import sglang.srt.layers.attention.fla.chunk as chunk_module
        import sglang.srt.layers.attention.fla.fused_recurrent as fused_recurrent_module

        # Import our bridge functions
        from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge
        from sglang_fl.dispatch.bridge.fla_fused_recurrent import (
            fused_recurrent_gated_delta_rule_bridge,
        )
        from sglang_fl.dispatch.bridge.fla_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode_bridge,
        )

        originals = {
            "chunk_gated_delta_rule": chunk_module.chunk_gated_delta_rule,
            "fused_recurrent_gated_delta_rule": (
                fused_recurrent_module.fused_recurrent_gated_delta_rule
            ),
            "fused_recurrent_gated_delta_rule_packed_decode": (
                fused_recurrent_module.fused_recurrent_gated_delta_rule_packed_decode
            ),
        }
        for name, fn in originals.items():
            _originals.setdefault(name, fn)

        bridges = {
            "chunk_gated_delta_rule": chunk_gated_delta_rule_bridge,
            "fused_recurrent_gated_delta_rule": (
                fused_recurrent_gated_delta_rule_bridge
            ),
            "fused_recurrent_gated_delta_rule_packed_decode": (
                fused_recurrent_gated_delta_rule_packed_decode_bridge
            ),
        }

        if vendor_only:
            from sglang_fl.dispatch import get_default_manager

            manager = get_default_manager()
            selected = _select_fla_implementations(
                _originals,
                bridges,
                vendor_only=True,
                is_vendor_available=manager.has_available_vendor,
            )
        else:
            selected = _select_fla_implementations(
                _originals, bridges, vendor_only=False
            )

        chunk_module.chunk_gated_delta_rule = selected["chunk_gated_delta_rule"]
        fused_recurrent_module.fused_recurrent_gated_delta_rule = selected[
            "fused_recurrent_gated_delta_rule"
        ]
        fused_recurrent_module.fused_recurrent_gated_delta_rule_packed_decode = (
            selected["fused_recurrent_gated_delta_rule_packed_decode"]
        )

        # Also patch the imports in gdn_triton.py (the actual call site)
        try:
            import sglang.srt.layers.attention.linear.kernels.gdn_triton as gdn_triton

            gdn_triton.chunk_gated_delta_rule = selected["chunk_gated_delta_rule"]
            gdn_triton.fused_recurrent_gated_delta_rule_packed_decode = selected[
                "fused_recurrent_gated_delta_rule_packed_decode"
            ]
            logger.info("Patched FLA functions in gdn_triton.py")
        except Exception as e:
            logger.warning(f"Failed to patch gdn_triton.py: {e}")

        dispatched = [name for name, fn in selected.items() if fn is bridges[name]]
        retained = [name for name in bridges if name not in dispatched]
        logger.info(
            "Configured FLA functions: dispatch=%s, sglang_original=%s",
            dispatched,
            retained,
        )

        return _originals

    except Exception as e:
        logger.error(f"Failed to patch FLA functions: {e}")
        return None
