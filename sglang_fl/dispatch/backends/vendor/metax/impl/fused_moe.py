# MetaX FusedMoE implementations.

from __future__ import annotations

import torch


def fused_moe_metax(obj, layer: torch.nn.Module, dispatch_output):
    from sglang_fl.dispatch.backends.vendor.cuda.impl.fused_moe import fused_moe_cuda

    return fused_moe_cuda(obj, layer, dispatch_output)
