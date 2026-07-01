# TXDA TopK operator implementation.
#
# On TXDA, no Triton kernels are available.  For the standard MoE runner we
# delegate to ``select_experts`` (torch-native path).  When the downstream
# runner is ``triton_kernel`` we must return ``TritonKernelTopKOutput``.
#
# ``routing_torch`` (from triton_kernels) is a pure-PyTorch reference
# implementation, but importing the ``triton_kernels.routing`` module can
# fail on TXDA because of a missing ``triton.language.target_info`` dependency.
# We therefore inline the necessary pure-PyTorch logic directly below.

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# TXDA TopK dispatch entry point
# ---------------------------------------------------------------------------


def topk_txda(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    from sglang.srt.layers.moe.topk import (
        select_experts,
    )
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    topk_cfg = obj.topk_config
    runner_backend = get_moe_runner_backend()

    # Standard path: torch-native select_experts returning StandardTopKOutput.
    topk_cfg.torch_native = True
    return select_experts(
        hidden_states=hidden_states,
        layer_id=obj.layer_id,
        router_logits=router_logits,
        topk_config=topk_cfg,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
