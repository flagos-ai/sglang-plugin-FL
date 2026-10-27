# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CoreX compatibility overlay, imported only for the Iluvatar vendor."""

from .patches.awq_dequantize import patch_awq_dequantize
from .patches.clamp_position import patch_clamp_position
from .patches.legacy_gpu_gate import patch_legacy_gpu_gate
from .patches.speculative_tree import patch_speculative_tree
from .patches.triton_pdl import patch_triton_pdl_intrinsics

_patches_applied = False


def apply_iluvatar_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return

    # SGLang imports and scans CUDA-flavoured kernels in the following patches.
    # PDL symbols and its false capability query must be installed first.
    patch_triton_pdl_intrinsics()
    patch_clamp_position()
    patch_legacy_gpu_gate()
    patch_speculative_tree()
    patch_awq_dequantize()
    _patches_applied = True


apply_iluvatar_patches()
