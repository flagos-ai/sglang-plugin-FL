# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Use upstream Triton dequantization for Iluvatar's ordinary AWQ linear path."""

import importlib
import logging

logger = logging.getLogger(__name__)
_applied = False
_TARGET_MODULE = "sglang.srt.hardware_backend.gpu.quantization.awq_kernels"
_TRITON_MODULE = "sglang.kernels.ops.quantization.awq_triton"


def patch_awq_dequantize() -> None:
    global _applied
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    if info is None or info.vendor_name != "iluvatar" or _applied:
        return

    module = importlib.import_module(_TARGET_MODULE)
    triton_module = importlib.import_module(_TRITON_MODULE)
    original = getattr(module, "awq_dequantize", None)
    replacement = getattr(triton_module, "awq_dequantize_triton", None)
    if not callable(original) or not callable(replacement):
        raise RuntimeError("Iluvatar requires SGLang's AWQ dequantization bindings")

    # AWQLinearKernel.apply resolves this module global at each call. Keep the
    # original matmul, optional bias and reshape; leave Marlin/MoE and the
    # registered torch custom op unchanged.
    module.awq_dequantize = replacement
    _applied = True
    logger.info("Iluvatar: ordinary AWQ linear dequantization uses upstream Triton")
