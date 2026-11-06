"""GCU int32 inputs for native multimodal isin, matching the prior adaptation."""

import torch


def multimodal_mask_gcu(original, input_ids, placeholder_tensor):
    if input_ids.device.type == "gcu":
        return original(
            input_ids.to(torch.int32),
            placeholder_tensor.to(torch.int32),
        )
    return original(input_ids, placeholder_tensor)


def patch_multimodal_mask():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.managers.mm_schedule._get_multimodal_mask",
        multimodal_mask_gcu,
        HookType.AROUND,
    )
