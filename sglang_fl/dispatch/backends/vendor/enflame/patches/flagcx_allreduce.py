"""Use FP32 accumulation for GCU BF16 FlagCX all-reduce."""

import torch


def all_reduce_gcu(original, comm, input_):
    if (
        comm.disabled
        or comm.device.type != "gcu"
        or input_.dtype != torch.bfloat16
    ):
        return original(comm, input_)

    # BF16 collective rounding can depend on the total message length.
    reduced = original(comm, input_.float())
    input_.copy_(reduced.to(input_.dtype))
    return input_


def patch_flagcx_allreduce():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang_fl.distributed.device_communicators.flagcx.FlagCXCommunicator.all_reduce",
        all_reduce_gcu,
        HookType.AROUND,
    )
