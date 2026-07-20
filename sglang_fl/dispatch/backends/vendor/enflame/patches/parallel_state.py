import torch
from sglang.srt.distributed.parallel_state import _groups

def gcu_reg_all_gather_into_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_gather_into_tensor(output, input)

def patch_parallel_state():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    server_args_class = "sglang.srt.distributed.parallel_state"
    HookRegistry.register(f"{server_args_class}.reg_all_gather_into_tensor", gcu_reg_all_gather_into_tensor, HookType.REPLACE)