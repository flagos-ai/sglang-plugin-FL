# Copyright (c) 2026 BAAI. All rights reserved.
"""Monkey-patch SGLang FLA functions to use dispatch mechanism."""

import importlib
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
        # SGLang 0.5.16 moved FLA kernels into the unified kernels namespace.
        # Keep the old path for the still-pinned non-CUDA environments.
        try:
            chunk_module = importlib.import_module(
                "sglang.kernels.ops.attention.fla.chunk"
            )
            fused_recurrent_module = importlib.import_module(
                "sglang.kernels.ops.attention.fla.fused_recurrent"
            )
        except ImportError:
            chunk_module = importlib.import_module(
                "sglang.srt.layers.attention.fla.chunk"
            )
            fused_recurrent_module = importlib.import_module(
                "sglang.srt.layers.attention.fla.fused_recurrent"
            )

        # Import our bridge functions
        from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge
        from sglang_fl.dispatch.bridge.fla_fused_recurrent import (
            fused_recurrent_gated_delta_rule_bridge,
        )
        from sglang_fl.dispatch.bridge.fla_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode_bridge,
        )

        # Save original functions before patching
        _originals.setdefault(
            "chunk_gated_delta_rule", chunk_module.chunk_gated_delta_rule
        )
        _originals.setdefault(
            "fused_recurrent_gated_delta_rule",
            fused_recurrent_module.fused_recurrent_gated_delta_rule,
        )
        _originals.setdefault(
            "fused_recurrent_gated_delta_rule_packed_decode",
            fused_recurrent_module.fused_recurrent_gated_delta_rule_packed_decode,
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
