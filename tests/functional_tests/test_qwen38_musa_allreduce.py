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
"""Run with torchrun --standalone --nproc_per_node=8 -m pytest <this file>."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_musa")
from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator  # noqa: E402
from sglang_fl.distributed.device_communicators.musa_custom_ar import (
    MusaCustomAllreduce,
)  # noqa: E402

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8,
    reason="requires eight local MUSA ranks launched by torchrun",
)


@pytest.fixture(scope="module")
def communication():
    rank = int(os.environ["LOCAL_RANK"])
    torch.musa.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    device = torch.device("musa", rank)
    # Model initialization sets the default device; control pointer tables
    # must nevertheless stay on CPU.
    with torch.device(device):
        communicator = MusaCustomAllreduce(dist.group.WORLD, device)
    assert communicator.signal_ptrs.device.type == "cpu"
    assert communicator.rank_data.device.type == "cpu"
    yield communicator
    communicator.close()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def flagcx(communication):
    communicator = FlagCXCommunicator(communication.group, communication.device)
    assert communicator.available and not communicator.disabled
    yield communicator
    del communicator


def test_selection_rejects_misaligned_contiguous_view(communication):
    comm = communication
    storage = torch.empty(64 * 2560 + 1, device=comm.device, dtype=torch.bfloat16)
    view = storage[1:].view(64, 2560)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    with comm.capture():
        assert not comm.should_custom_ar(view)


def expected_input(rows, hidden, rank, phase):
    return (
        (torch.arange(rows * hidden).reshape(rows, hidden) + rank + phase) % 15 - 7
    ).bfloat16()


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("hidden", [2560, 10240])
def test_eager_and_recaptured_graphs(communication, flagcx, rows, hidden):
    comm = communication
    for generation in range(2):
        partial = torch.empty(rows, hidden, device=comm.device, dtype=torch.bfloat16)
        for phase in range(3):
            data = expected_input(rows, hidden, comm.rank, phase)
            expected = sum(
                expected_input(rows, hidden, peer, phase).float() for peer in range(8)
            ).bfloat16()
            partial.copy_(data)
            actual = comm.custom_all_reduce(partial)
            torch.musa.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
            for inplace in (False, True):
                flagcx._use_inplace_all_reduce = inplace
                baseline = data.to(comm.device)
                alias = flagcx.all_reduce(baseline)
                assert alias is baseline
                torch.musa.synchronize()
                torch.testing.assert_close(baseline.cpu(), expected, atol=0, rtol=0)
                torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        with comm.capture():
            comm.custom_all_reduce(partial)
            torch.musa.synchronize()
            dist.barrier()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph):
                actual = comm.custom_all_reduce(partial)
        for phase in range(3):
            partial.copy_(expected_input(rows, hidden, comm.rank, phase))
            actual.fill_(float("nan"))
            graph.replay()
            torch.musa.synchronize()
            expected = sum(
                expected_input(rows, hidden, peer, phase).float() for peer in range(8)
            ).bfloat16()
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        del graph, actual, partial
        comm.reset_graph_buffers()


@pytest.mark.parametrize("rows", [1024, 16384])
@pytest.mark.parametrize("hidden", [2560, 10240])
def test_flagcx_prefill_inplace_alias(flagcx, rows, hidden):
    flagcx._use_inplace_all_reduce = True
    partial = torch.empty(rows, hidden, device=flagcx.device, dtype=torch.bfloat16)
    for phase in range(2):
        partial.fill_(flagcx.rank + 1 + phase)
        result = flagcx.all_reduce(partial)
        assert result is partial
        torch.musa.synchronize()
        assert torch.all(result == 36 + 8 * phase).item()


def test_group_destroy_and_rebuild(communication):
    comm = communication
    comm.close()
    for generation in range(2):
        group = dist.new_group(list(range(8)), backend="gloo")
        replacement = MusaCustomAllreduce(group, comm.device)
        partial = torch.full(
            (64, 2560),
            comm.rank + 1 + generation,
            device=comm.device,
            dtype=torch.bfloat16,
        )
        actual = replacement.custom_all_reduce(partial)
        torch.musa.synchronize()
        torch.testing.assert_close(
            actual.cpu(),
            torch.full_like(actual.cpu(), 36 + 8 * generation),
            atol=0,
            rtol=0,
        )
        replacement.close()
        dist.destroy_process_group(group)
