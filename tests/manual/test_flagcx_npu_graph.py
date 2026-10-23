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

"""Manual four-rank Ascend FlagCX graph/storage regression (not a pytest test).

Run in the prepared Ascend runtime, with four free visible NPU devices:
  torchrun --standalone --nproc-per-node=4 tests/manual/test_flagcx_npu_graph.py

Uses the installed production communicator, no diagnostic helper or fallback.
Captures several sizes into one shared pool, then alternates graph replay,
eager all-reduce, and an independent 256-int64 metadata oracle. CPU reads occur
only outside capture. Keep graphs and their input/output tensors alive, but
do not retain the communicator's private temporary output.
"""

import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path


def emit(event, **fields):
    print(
        json.dumps(
            {"event": event, "rank": int(os.environ.get("RANK", "-1")), **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 8, 1])
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if min(args.rounds, args.timeout_seconds, *args.sizes) < 1:
        parser.error("rounds, sizes, and timeout must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be distinct")

    import torch
    import torch.distributed as dist
    import torch_npu

    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    rank, local_rank, world = (
        int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    if world != 4:
        raise RuntimeError("This regression requires exactly four torchrun ranks")
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dtype = torch.bfloat16
    dist.init_process_group(
        "gloo", timeout=datetime.timedelta(seconds=args.timeout_seconds)
    )
    comm = FlagCXCommunicator(group=dist.group.WORLD, device=device)
    if not comm.available or comm.disabled:
        raise RuntimeError(
            "Production FlagCX communicator unavailable; fallback forbidden"
        )
    source = Path(inspect.getfile(FlagCXCommunicator)).resolve()
    emit(
        "START",
        rounds=args.rounds,
        sizes=args.sizes,
        world_size=world,
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        communicator_file=str(source),
        communicator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        environment={
            key: os.environ.get(key)
            for key in (
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_LAUNCH_BLOCKING",
                "FLAGCX_PATH",
                "TASK_QUEUE_ENABLE",
                "HCCL_BUFFSIZE",
                "HCCL_OP_EXPANSION_MODE",
                "STREAMS_PER_DEVICE",
            )
        },
    )

    def forward(input_tensor):
        produced = torch.add(input_tensor, 1)
        if comm.all_reduce(produced) is not produced:
            raise AssertionError("Production all_reduce must return its input in-place")
        scaled = torch.mul(produced, 2)
        return produced, scaled, torch.add(scaled, 1)

    def oracle(shape, offset):
        expected = 2 * (world * (world + 1) // 2 + world * (offset + 1)) + 1
        return torch.full(shape, expected, dtype=dtype)

    def check(actual, expected, label):
        if not torch.equal(actual, expected):
            bad = actual.ne(expected)
            indices = bad.flatten().nonzero().flatten()[:8].tolist()
            emit(
                "FAIL_VALUES",
                label=label,
                mismatch_count=int(bad.sum().item()),
                first_bad_indices=indices,
                actual=[repr(value) for value in actual.flatten()[indices].tolist()],
                expected=expected.flatten()[indices].tolist(),
            )
            raise AssertionError(f"{label}: exact CPU oracle mismatch")

    main_stream = torch.npu.current_stream()
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(main_stream)
    pool = torch.npu.graph_pool_handle()
    cases, warmup_refs = [], []
    for size in args.sizes:
        shape = (size, 2048)
        static_input = torch_npu.npu_format_cast(
            torch.full(shape, rank + 1, dtype=dtype, device=device), 2
        )
        if int(torch_npu.get_npu_format(static_input)) != 2:
            raise RuntimeError("Requested NPU storage format 2 was not realized")
        capture_stream.wait_stream(main_stream)
        with torch.npu.stream(capture_stream):
            for _ in range(3):
                warmup_refs.append(forward(static_input))
        main_stream.wait_stream(capture_stream)
        check(warmup_refs[-1][-1].cpu(), oracle(shape, 0), f"warmup size={size}")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph, pool=pool, stream=capture_stream, auto_dispatch_capture=True
        ):
            captured = forward(static_input)
        main_stream.wait_stream(capture_stream)
        cases.append((size, shape, static_input, graph, captured))
        emit("CAPTURE_DONE", size=size, auto_dispatch_capture=True)

    for round_index in range(args.rounds):
        # Captures share a pool but never execute concurrently. Check each
        # result before another bucket may overwrite shared graph storage.
        ordered = cases if round_index % 2 == 0 else reversed(cases)
        for size, shape, static_input, graph, captured in ordered:
            offset = round_index % 7
            static_input.fill_(rank + 1 + offset)
            graph.replay()
            expected = oracle(shape, offset)
            check(
                captured[-1].cpu(), expected, f"graph round={round_index} size={size}"
            )
            metadata = torch.add(
                torch.arange(256, dtype=torch.int64, device=device), round_index
            )
            eager_input = torch.full(
                shape, rank + 1 + offset, dtype=dtype, device=device
            )
            eager_refs = forward(eager_input)
            check(
                eager_refs[-1].cpu(), expected, f"eager round={round_index} size={size}"
            )
            check(
                metadata.cpu(),
                torch.arange(256, dtype=torch.int64) + round_index,
                f"metadata round={round_index} size={size}",
            )
            emit("ROUND_PASS", round=round_index, size=size)
            del metadata, eager_input, eager_refs, expected

    completed = {
        "rank": rank,
        "rounds": args.rounds,
        "sizes": args.sizes,
        "status": "PASS",
    }
    all_completed = [None] * world
    dist.all_gather_object(all_completed, completed)
    if any(
        item["status"] != "PASS"
        or item["rounds"] != args.rounds
        or item["sizes"] != args.sizes
        for item in all_completed
    ):
        raise AssertionError("Not all ranks completed every requested check")
    emit("SUMMARY", status="PASS", per_rank=all_completed)
    # comm, cases and warmup_refs remain live through the final all-rank oracle.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
