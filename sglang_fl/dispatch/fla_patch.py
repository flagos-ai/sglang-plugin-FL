# Copyright (c) 2026 BAAI. All rights reserved.
"""Monkey-patch SGLang FLA functions to use dispatch mechanism."""

import logging

logger = logging.getLogger(__name__)

# Store original (unpatched) FLA functions so backends can call them without recursion.
_originals = {}


def get_original(name: str):
    """Get the original (unpatched) FLA function by name."""
    return _originals.get(name)


def patch_fla_functions():
    """
    Replace SGLang's FLA module-level functions with dispatch bridges.

    This allows FLA ops to go through the dispatch system like other fused ops.
    """
    try:
        # Import SGLang FLA modules
        try:
            import sglang.srt.layers.attention.fla.chunk as chunk_module
            import sglang.srt.layers.attention.fla.fused_recurrent as fused_recurrent_module
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("sglang.srt.layers.attention.fla"):
                raise
            import sglang.kernels.ops.attention.fla.chunk as chunk_module
            import sglang.kernels.ops.attention.fla.fused_recurrent as fused_recurrent_module
            from sglang_fl.utils import get_device_info

            info = get_device_info()
            if info is not None and info.vendor_name == "mthreads":
                # Current MUSA backend delegates to these originals anyway.
                # The new chunk API adds in-place/output-index contracts which
                # the legacy bridge does not expose. Keep native direct calls.
                for name, module in (
                    ("chunk_gated_delta_rule", chunk_module),
                    ("fused_recurrent_gated_delta_rule", fused_recurrent_module),
                    ("fused_recurrent_gated_delta_rule_packed_decode", fused_recurrent_module),
                ):
                    _originals[name] = getattr(module, name)
                logger.info("MUSA: retained relocated native FLA kernels and their current signatures")
                return _originals

        # Import our bridge functions
        from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge
        from sglang_fl.dispatch.bridge.fla_fused_recurrent import (
            fused_recurrent_gated_delta_rule_bridge,
        )
        from sglang_fl.dispatch.bridge.fla_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode_bridge,
        )

        # Save original functions before patching
        _originals["chunk_gated_delta_rule"] = chunk_module.chunk_gated_delta_rule
        _originals["fused_recurrent_gated_delta_rule"] = (
            fused_recurrent_module.fused_recurrent_gated_delta_rule
        )
        _originals["fused_recurrent_gated_delta_rule_packed_decode"] = (
            fused_recurrent_module.fused_recurrent_gated_delta_rule_packed_decode
        )

        # Replace with bridge functions
        chunk_module.chunk_gated_delta_rule = chunk_gated_delta_rule_bridge
        fused_recurrent_module.fused_recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule_bridge
        )
        fused_recurrent_module.fused_recurrent_gated_delta_rule_packed_decode = (
            fused_recurrent_gated_delta_rule_packed_decode_bridge
        )

        # Also patch the imports in gdn_triton.py (the actual call site)
        try:
            import sglang.srt.layers.attention.linear.kernels.gdn_triton as gdn_triton

            gdn_triton.chunk_gated_delta_rule = chunk_gated_delta_rule_bridge
            gdn_triton.fused_recurrent_gated_delta_rule_packed_decode = (
                fused_recurrent_gated_delta_rule_packed_decode_bridge
            )
            logger.info("Patched FLA functions in gdn_triton.py")
        except Exception as e:
            logger.warning(f"Failed to patch gdn_triton.py: {e}")

        logger.info("Successfully patched FLA functions for dispatch")

        return _originals

    except Exception as e:
        logger.error(f"Failed to patch FLA functions: {e}")
        return None
