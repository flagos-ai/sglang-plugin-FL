# MUSA FusedMoE operator implementation.

from __future__ import annotations

import torch


def fused_moe_musa(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    return obj.forward_musa(layer, dispatch_output)

