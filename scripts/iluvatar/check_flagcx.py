#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun; verify native FlagCX collectives without fallback."""

from datetime import timedelta
import os


def main():
    import torch
    import torch.distributed as dist

    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    if not hasattr(torch, "corex") or not torch.cuda.is_available():
        raise RuntimeError("This probe requires visible CoreX GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size < 2:
            raise RuntimeError("Run this probe with at least two ranks")
        comm = FlagCXCommunicator(dist.group.WORLD, device)
        if not comm.available or comm.disabled:
            raise RuntimeError(
                "Native FlagCX did not initialize; fallback is forbidden"
            )
        values = torch.full((8,), rank + 1.0, device=device)
        result = comm.all_reduce(values)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            result, torch.full_like(values, size * (size + 1) / 2)
        )
        source = torch.full((8,), float(rank), device=device)
        gathered = torch.empty(size, 8, device=device)
        comm.all_gather(gathered, source)
        torch.cuda.synchronize()
        expected = torch.arange(size, device=device, dtype=source.dtype)[
            :, None
        ].expand(size, 8)
        torch.testing.assert_close(gathered, expected)
        dist.barrier()
        if rank == 0:
            print(f"Native FlagCX all_reduce and all_gather passed on {size} ranks")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
