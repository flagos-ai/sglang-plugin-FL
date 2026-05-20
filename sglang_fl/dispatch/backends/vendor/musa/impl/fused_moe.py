# MUSA FusedMoE operator implementation.
#
# Mirrors sglang's UnquantizedFusedMoEMethod.forward_musa() which explicitly
# delegates to forward_cuda() (srt/layers/quantization/unquant.py).
# forward_cuda() uses Triton fused MoE kernels that run on MUSA.

from __future__ import annotations

import torch


def fused_moe_musa(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    return obj.forward_cuda(layer, dispatch_output)

