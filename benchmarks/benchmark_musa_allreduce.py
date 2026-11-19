# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TP8 message benchmark: torchrun --standalone --nproc_per_node=8 <this file>."""

import argparse
import contextlib
import datetime
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator
from sglang_fl.distributed.device_communicators.musa_custom_ar import (
    MusaCustomAllreduce,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64, choices=(1, 7, 64, 256))
    parser.add_argument("--hidden", type=int, default=2560, choices=(2560, 10240))
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.musa.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    device = torch.device("musa", rank)
    custom = MusaCustomAllreduce(dist.group.WORLD, device, max_size=8 * 1024 * 1024)
    flagcx = FlagCXCommunicator(dist.group.WORLD, device)
    partial = torch.zeros(args.rows, args.hidden, device=device, dtype=torch.bfloat16)
    for backend in ("flagcx_copy", "flagcx_inplace", "custom"):
        flagcx._use_inplace_all_reduce = backend == "flagcx_inplace"

        def operation():
            if backend == "custom":
                partial.copy_(custom.custom_all_reduce(partial))
            else:
                flagcx.all_reduce(partial)

        capture = custom.capture() if backend == "custom" else contextlib.nullcontext()
        with capture:
            operation()
            torch.musa.synchronize()
            dist.barrier()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph):
                for _ in range(32):
                    operation()
        for _ in range(3):
            graph.replay()
        torch.musa.synchronize()
        dist.barrier()
        times = []
        for _ in range(3):
            begin = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) * 1000 / 320)
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        backend=backend,
                        rows=args.rows,
                        hidden=args.hidden,
                        bytes=partial.numel() * partial.element_size(),
                        median_us=statistics.median(times),
                        trials_us=times,
                    )
                ),
                flush=True,
            )
        del graph
        custom.reset_graph_buffers()
    custom.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
