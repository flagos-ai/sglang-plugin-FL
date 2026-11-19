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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang_fl.distributed.communicator import CommunicatorFL
from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator
from sglang_fl.distributed.device_communicators.musa_custom_ar import (
    MusaCustomAllreduce,
)


def test_collective_failure_propagates_without_retry(monkeypatch):
    communicator = CommunicatorFL.__new__(CommunicatorFL)
    failed = Mock(side_effect=RuntimeError("collective failed"))
    communicator._flagcx_comm = SimpleNamespace(disabled=False, all_reduce=failed)
    other = Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", other)
    with pytest.raises(RuntimeError, match="collective failed"):
        communicator.all_reduce(torch.zeros(8))
    assert failed.call_count == 1
    other.assert_not_called()


def test_flagcx_communicator_return_alias():
    communicator = CommunicatorFL.__new__(CommunicatorFL)
    communicator._flagcx_comm = SimpleNamespace(disabled=False, all_reduce=lambda x: x)
    data = torch.ones(8)
    assert communicator.all_reduce(data) is data


def test_musa_inplace_pointer_and_strided_non_hit():
    communicator = FlagCXCommunicator.__new__(FlagCXCommunicator)
    communicator.device = SimpleNamespace(type="musa", index=0)
    communicator._use_inplace_all_reduce = True
    communicator._buffer_type = int
    communicator._dtype_enum = SimpleNamespace(from_torch=lambda x: x)
    communicator._redop_enum = SimpleNamespace(from_torch=lambda x: x)
    communicator.comm = object()
    communicator._get_stream = lambda: None
    launch = Mock()
    communicator.flagcx = SimpleNamespace(flagcxAllReduce=launch)
    tensor = torch.ones(7, 2560, dtype=torch.bfloat16)
    assert communicator._flagcx_all_reduce(tensor) is tensor
    args = launch.call_args.args
    assert args[0] == args[1] == tensor.data_ptr()
    launch.reset_mock()
    view = tensor.t()
    result = communicator._flagcx_all_reduce(view)
    assert result is not view
    assert launch.call_args.args[0] != launch.call_args.args[1]
    for unsupported in (tensor.float(), tensor[:, :8].contiguous()):
        assert communicator._flagcx_all_reduce(unsupported) is not unsupported


@pytest.mark.parametrize(
    "dtype,contiguous,hidden,device",
    [
        (torch.float32, True, 2560, "musa"),
        (torch.bfloat16, False, 2560, "musa"),
        (torch.bfloat16, True, 3, "musa"),
        (torch.bfloat16, True, 65536, "musa"),
        (torch.bfloat16, True, 2560, "cpu"),
    ],
)
def test_custom_ar_selection_rejects_unvalidated_inputs(
    dtype, contiguous, hidden, device
):
    communicator = MusaCustomAllreduce.__new__(MusaCustomAllreduce)
    communicator.device = SimpleNamespace(type="musa", index=0)
    communicator.max_size = 2 * 1024 * 1024
    communicator._capturing = True
    tensor = SimpleNamespace(
        device=SimpleNamespace(type=device, index=0),
        dtype=dtype,
        is_contiguous=lambda: contiguous,
        ndim=2,
        shape=(64, hidden),
        numel=lambda: 64 * hidden,
        element_size=lambda: 2,
        data_ptr=lambda: 0x1000,
    )
    assert not communicator.should_custom_ar(tensor)


def registered_all_reduce_hook(monkeypatch):
    import sglang_fl
    from sglang.srt.plugins.hook_registry import HookRegistry

    hooks = {}
    monkeypatch.setattr(
        HookRegistry, "register", lambda name, fn, kind: hooks.__setitem__(name, fn)
    )
    sglang_fl._setup_communicator_hooks()
    return hooks["sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce"]


@pytest.mark.parametrize(
    "selected,fails", [(True, False), (False, False), (True, True)]
)
def test_model_group_dispatch_contract(monkeypatch, selected, fails):
    hook = registered_all_reduce_hook(monkeypatch)
    tensor = torch.ones(2, 8)
    custom = SimpleNamespace(
        should_custom_ar=lambda x: selected,
        custom_all_reduce=Mock(side_effect=RuntimeError("execution failed"))
        if fails
        else Mock(return_value=torch.full_like(tensor, 8)),
    )
    flagcx = SimpleNamespace(disabled=False, all_reduce=Mock(return_value=tensor))
    group = SimpleNamespace(_fl_qwen38_custom_ar=custom, fl_communicator=flagcx)
    original = Mock()
    if fails:
        with pytest.raises(RuntimeError, match="execution failed"):
            hook(original, group, tensor)
        flagcx.all_reduce.assert_not_called()
    else:
        assert hook(original, group, tensor) is tensor
        assert flagcx.all_reduce.call_count == int(not selected)
        if selected:
            torch.testing.assert_close(
                tensor, torch.full_like(tensor, 8), atol=0, rtol=0
            )
    assert custom.custom_all_reduce.call_count == int(selected)
    original.assert_not_called()


def test_group_dispatch_preserves_non_target_path(monkeypatch):
    hook = registered_all_reduce_hook(monkeypatch)
    tensor = torch.ones(8)
    group = SimpleNamespace(fl_communicator=None)
    original = Mock(return_value=tensor)
    assert hook(original, group, tensor) is tensor
    original.assert_called_once_with(group, tensor)


def test_capture_selection_requires_registered_group(monkeypatch):
    comm = MusaCustomAllreduce.__new__(MusaCustomAllreduce)
    comm.device = SimpleNamespace(type="musa", index=0)
    comm.max_size = 2 * 1024 * 1024
    comm._capturing = False
    tensor = SimpleNamespace(
        device=comm.device,
        dtype=torch.bfloat16,
        is_contiguous=lambda: True,
        ndim=2,
        shape=(64, 2560),
        numel=lambda: 64 * 2560,
        element_size=lambda: 2,
        data_ptr=lambda: 0x1000,
    )
    assert not comm.should_custom_ar(tensor)
    comm._capturing = True
    assert comm.should_custom_ar(tensor)
    tensor.data_ptr = lambda: 0x1002
    assert not comm.should_custom_ar(tensor)


def test_destroy_releases_custom_group_before_framework():
    from sglang_fl.models.qwen4_communication import _destroy

    order = []
    group = SimpleNamespace(
        _fl_qwen38_custom_ar=SimpleNamespace(close=lambda: order.append("close"))
    )
    _destroy(lambda self: order.append("destroy"), group)
    assert order == ["close", "destroy"]
    assert group.ca_comm is None and group._fl_qwen38_custom_ar is None


@pytest.mark.parametrize("rows", [1, 7, 32])
def test_custom_ar_keeps_smaller_batches_on_flagcx(rows):
    comm = MusaCustomAllreduce.__new__(MusaCustomAllreduce)
    comm.device = SimpleNamespace(type="musa", index=0)
    comm.max_size = 2 * 1024 * 1024
    comm._capturing = True
    tensor = SimpleNamespace(
        device=comm.device,
        dtype=torch.bfloat16,
        is_contiguous=lambda: True,
        ndim=2,
        shape=(rows, 2560),
        numel=lambda: rows * 2560,
        element_size=lambda: 2,
        data_ptr=lambda: 0x1000,
    )
    assert not comm.should_custom_ar(tensor)


@pytest.mark.parametrize("unsupported", ["remote", "chip", "peer"])
def test_prepare_rejects_unsupported_topology_before_allocation(
    monkeypatch, unsupported
):
    from sglang_fl.models import qwen4_communication as communication
    from sglang_fl.distributed.device_communicators import musa_custom_ar

    group = SimpleNamespace(
        world_size=8,
        device=SimpleNamespace(type="musa", index=0),
        cpu_group=object(),
        use_custom_allreduce=True,
        ca_comm=None,
    )
    monkeypatch.setattr(communication, "groups", lambda: (group,))
    monkeypatch.setattr(communication.socket, "gethostname", lambda: "local")
    monkeypatch.setattr(
        torch.musa,
        "get_device_properties",
        lambda device: SimpleNamespace(name="MTT S5000"),
    )
    monkeypatch.setattr(torch.musa, "can_device_access_peer", lambda device, peer: True)

    def gather(output, value, group):
        if isinstance(value, tuple):
            output[:] = [("local", i, "MTT S5000") for i in range(8)]
            if unsupported == "remote":
                output[7] = ("other-host", 7, "MTT S5000")
            elif unsupported == "chip":
                output[7] = ("local", 7, "unvalidated-chip")
        else:
            output[:] = [True] * 8
            if unsupported == "peer":
                output[7] = False

    monkeypatch.setattr(communication.dist, "all_gather_object", gather)

    class NoAllocation:
        def __init__(self, *args):
            pytest.fail("Unsupported topology must not allocate IPC buffers")

    monkeypatch.setattr(musa_custom_ar, "MusaCustomAllreduce", NoAllocation)
    communication.prepare()
    assert group.ca_comm is None


def test_prepare_respects_framework_custom_allreduce_disable(monkeypatch):
    from sglang_fl.models import qwen4_communication as communication

    group = SimpleNamespace(
        world_size=8,
        device=SimpleNamespace(type="musa", index=0),
        use_custom_allreduce=False,
        ca_comm=None,
    )
    monkeypatch.setattr(communication, "groups", lambda: (group,))
    properties = Mock(
        side_effect=AssertionError("Disabled backend must not initialize")
    )
    monkeypatch.setattr(torch.musa, "get_device_properties", properties)
    communication.prepare()
    properties.assert_not_called()
    assert group.ca_comm is None


def test_prepare_enables_inplace_only_for_validated_model_group(monkeypatch):
    from sglang_fl.models import qwen4_communication as communication
    from sglang_fl.distributed.device_communicators import musa_custom_ar

    flagcx = SimpleNamespace(disabled=False, _use_inplace_all_reduce=False)
    group = SimpleNamespace(
        world_size=8,
        device=SimpleNamespace(type="musa", index=0),
        cpu_group=object(),
        use_custom_allreduce=True,
        ca_comm=None,
        fl_communicator=SimpleNamespace(_flagcx_comm=flagcx),
    )
    monkeypatch.setattr(communication, "groups", lambda: (group,))
    monkeypatch.setattr(communication.socket, "gethostname", lambda: "local")
    monkeypatch.setattr(
        torch.musa,
        "get_device_properties",
        lambda device: SimpleNamespace(name="MTT S5000"),
    )
    monkeypatch.setattr(torch.musa, "can_device_access_peer", lambda device, peer: True)

    def gather(output, value, group):
        output[:] = (
            [("local", i, "MTT S5000") for i in range(8)]
            if isinstance(value, tuple)
            else [True] * 8
        )

    monkeypatch.setattr(communication.dist, "all_gather_object", gather)
    allocations = []

    class Prepared:
        def __init__(self, cpu_group, device):
            allocations.append((cpu_group, device))

    monkeypatch.setattr(musa_custom_ar, "MusaCustomAllreduce", Prepared)
    communication.prepare()
    communication.prepare()
    assert flagcx._use_inplace_all_reduce
    assert len(allocations) == 1
    assert group.ca_comm is group._fl_qwen38_custom_ar
