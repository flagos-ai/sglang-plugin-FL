"""GCU expert-index compatibility for reference TopK routing."""

import torch


def topk_torch_gcu(original, obj, hidden_states, router_logits, **kwargs):
    output = original(obj, hidden_states, router_logits, **kwargs)
    if hidden_states.device.type != "gcu":
        return output
    return output._replace(topk_ids=output.topk_ids.to(torch.int32))


def patch_topk():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang_fl.dispatch.backends.reference.impl.topk.topk_torch",
        topk_torch_gcu,
        HookType.AROUND,
    )
