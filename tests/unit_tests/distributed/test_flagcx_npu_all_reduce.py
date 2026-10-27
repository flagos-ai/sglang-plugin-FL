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

"""CPU contract tests for NPU FlagCX all-reduce stream and storage ordering.

These mocks verify scheduling contracts, not NPU graph or numerical correctness.
Each test loads an isolated source module and replaces only that module's Torch
reference: no global Torch methods, distributed state, or imports are replaced.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/distributed/device_communicators/flagcx.py"
)


@pytest.fixture
def flagcx_module():
    spec = importlib.util.spec_from_file_location("_isolated_flagcx_npu_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Device:
    def __init__(self, kind):
        self.type = kind

    def __str__(self):
        return f"{self.type}:0"


class _ForbiddenNPU:
    def __getattr__(self, name):
        raise AssertionError(f"Non-NPU all-reduce accessed torch.npu.{name}")


class _Stream:
    def __init__(self, runtime, name):
        self.runtime = runtime
        self.name = name

    def wait_stream(self, other):
        self.runtime.events.append((self.name, "wait", other.name))

    def synchronize(self):
        raise AssertionError("All-reduce must not add host stream synchronization")


class _Tensor:
    def __init__(self, runtime, name, pointer, *, numel=2048):
        self.runtime = runtime
        self.name = name
        self.pointer = pointer
        self.device = runtime.device
        self.dtype = "test-bfloat16"
        self._numel = numel

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self._numel

    def record_stream(self, stream):
        self.runtime.events.append((self.name, "record", stream.name))

    def copy_(self, other):
        self.runtime.events.append(
            ("copy", self.name, other.name, self.runtime.current.name)
        )
        return self


class _Runtime:
    def __init__(self, module, kind="npu"):
        self.module = module
        self.device = _Device(kind)
        self.events = []
        self.outputs = []
        self.native_calls = []
        self.stream_creations = 0
        self.active_device = None
        self.capturing = False
        self.native_error = None
        self.forward = _Stream(self, "forward")
        self.communication = _Stream(self, "comm")
        self.current = self.forward
        self.comm_handle = object()
        self.sum_op = object()
        self.library = SimpleNamespace(
            flagcxAllReduce=self.native_all_reduce,
            flagcxCommDestroy=lambda comm: None,
            flagcxGetUniqueId=lambda: SimpleNamespace(internal=[0] * 128),
            flagcxCommInitRank=self.init_comm,
        )
        self.dtype_enum = SimpleNamespace(from_torch=lambda dtype: ("dtype", dtype))
        self.redop_enum = SimpleNamespace(from_torch=lambda op: ("reduce", op))
        self.torch = SimpleNamespace(
            empty_like=self.empty_like,
            zeros=self.zeros,
            ByteTensor=lambda values: SimpleNamespace(tolist=lambda: list(values)),
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(SUM=self.sum_op)),
            cuda=SimpleNamespace(device=self.device_context),
            musa=SimpleNamespace(device=self.device_context),
            npu=SimpleNamespace(
                current_stream=self.current_stream,
                stream=self.stream_context,
                Stream=self.new_stream,
                device=self.device_context,
                is_current_stream_capturing=lambda: self.capturing,
                synchronize=self.forbidden_sync,
            ),
        )
        module.torch = self.torch

    def new_tensor(self, name="input", pointer=100, *, numel=2048):
        return _Tensor(self, name, pointer, numel=numel)

    def empty_like(self, tensor):
        self.events.append(("allocate", self.current.name, tensor.name))
        output = self.new_tensor(
            f"out{len(self.outputs)}", 200 + len(self.outputs), numel=tensor.numel()
        )
        self.outputs.append(output)
        return output

    def zeros(self, size, *, device):
        assert device is self.device
        self.events.append(("zeros", self.current.name))
        return self.new_tensor("warmup", numel=size)

    def current_stream(self, device=None):
        assert device is None or device is self.device
        self.events.append(("current", self.current.name))
        return self.current

    def new_stream(self):
        assert self.active_device is self.device
        self.stream_creations += 1
        self.events.append(("create_stream", self.current.name))
        return self.communication

    @contextmanager
    def stream_context(self, stream):
        previous = self.current
        self.events.append(("enter_stream", stream.name))
        self.current = stream
        try:
            yield
        finally:
            self.current = previous
            self.events.append(("exit_stream", stream.name))

    @contextmanager
    def device_context(self, device):
        previous = self.active_device
        self.active_device = device
        try:
            yield
        finally:
            self.active_device = previous

    def get_flagcx_stream(self):
        self.events.append(("get_flagcx_stream", self.current.name))
        return ("flagcx_stream", self.current.name)

    def init_comm(self, world_size, unique_id, rank):
        self.events.append(("init_comm", world_size, rank))
        return self.comm_handle

    def native_all_reduce(self, *args):
        self.events.append(("native", self.current.name))
        self.native_calls.append(args)
        if self.native_error is not None:
            raise self.native_error

    @staticmethod
    def forbidden_sync():
        raise AssertionError("All-reduce must not add device synchronization")

    def communicator(self):
        cls = self.module.FlagCXCommunicator
        comm = cls.__new__(cls)
        comm.disabled = False
        comm.available = True
        comm.device = self.device
        comm.comm = self.comm_handle
        comm.flagcx = self.library
        comm._stream_cache = {}
        comm._npu_comm_stream = (
            self.communication if self.device.type == "npu" else None
        )
        comm._buffer_type = lambda ptr=None: ("buffer", ptr)
        comm._dtype_enum = self.dtype_enum
        comm._redop_enum = self.redop_enum
        comm._get_stream = self.get_flagcx_stream
        return comm

    def expected_native_args(self, tensor, output, stream):
        return (
            ("buffer", tensor.data_ptr()),
            ("buffer", output.data_ptr()),
            tensor.numel(),
            ("dtype", tensor.dtype),
            ("reduce", self.sum_op),
            self.comm_handle,
            ("flagcx_stream", stream),
        )


def _npu_order(output="out0", caller="forward"):
    return [
        ("allocate", caller, "input"),
        ("current", caller),
        ("comm", "wait", caller),
        ("input", "record", "comm"),
        (output, "record", "comm"),
        ("enter_stream", "comm"),
        ("get_flagcx_stream", "comm"),
        ("native", "comm"),
        ("exit_stream", "comm"),
        (caller, "wait", "comm"),
    ]


@pytest.mark.parametrize("capturing", [False, True])
def test_npu_all_reduce_allocates_on_caller_and_forks_joins_without_sync(
    flagcx_module, capturing
):
    runtime = _Runtime(flagcx_module)
    runtime.capturing = capturing
    comm = runtime.communicator()
    tensor = runtime.new_tensor()

    output = comm._flagcx_all_reduce(tensor)

    assert output is runtime.outputs[0]
    assert output is not tensor
    assert runtime.events == _npu_order()
    assert runtime.native_calls == [
        runtime.expected_native_args(tensor, output, "comm")
    ]
    assert runtime.current is runtime.forward
    assert runtime.stream_creations == 0


def test_npu_reuses_precreated_stream_for_multiple_callers_and_capture_states(
    flagcx_module,
):
    runtime = _Runtime(flagcx_module)
    comm = runtime.communicator()
    tensor = runtime.new_tensor()
    for index, capturing in enumerate((False, True, True, False)):
        caller = _Stream(runtime, f"caller{index}")
        runtime.current = caller
        runtime.capturing = capturing
        runtime.events.clear()

        output = comm._flagcx_all_reduce(tensor)

        assert runtime.events == _npu_order(f"out{index}", caller.name)
        assert output is runtime.outputs[index]
        assert comm._npu_comm_stream is runtime.communication
        assert runtime.current is caller
    assert runtime.stream_creations == 0


@pytest.mark.parametrize("kind", ["cuda", "musa"])
def test_other_accelerators_keep_native_current_stream_path_without_npu_api(
    flagcx_module, kind
):
    runtime = _Runtime(flagcx_module, kind)
    runtime.torch.npu = _ForbiddenNPU()
    comm = runtime.communicator()
    tensor = runtime.new_tensor()

    output = comm._flagcx_all_reduce(tensor)

    assert output is runtime.outputs[0]
    assert runtime.events == [
        ("allocate", "forward", "input"),
        ("get_flagcx_stream", "forward"),
        ("native", "forward"),
    ]
    assert runtime.native_calls == [
        runtime.expected_native_args(tensor, output, "forward")
    ]


@pytest.mark.parametrize("kind", ["npu", "cuda", "musa"])
def test_public_all_reduce_keeps_input_identity_and_original_copyback(
    flagcx_module, kind
):
    runtime = _Runtime(flagcx_module, kind)
    if kind != "npu":
        runtime.torch.npu = _ForbiddenNPU()
    comm = runtime.communicator()
    tensor = runtime.new_tensor()

    returned = comm.all_reduce(tensor)

    assert returned is tensor
    assert runtime.events[-1] == ("copy", "input", "out0", "forward")
    assert len(runtime.native_calls) == 1
    if kind == "npu":
        assert runtime.events[:-1] == _npu_order()


def test_native_error_restores_caller_stream_and_is_not_swallowed(flagcx_module):
    runtime = _Runtime(flagcx_module)
    comm = runtime.communicator()
    runtime.native_error = RuntimeError("native FlagCX failure")

    with pytest.raises(RuntimeError, match="native FlagCX failure"):
        comm.all_reduce(runtime.new_tensor())

    assert runtime.current is runtime.forward
    assert runtime.events[-1] == ("exit_stream", "comm")
    assert not any(event[0] == "copy" for event in runtime.events)


def test_constructor_creates_npu_stream_before_eager_warmup(
    flagcx_module, monkeypatch, tmp_path
):
    runtime = _Runtime(flagcx_module)
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(
        flagcx_module,
        "_import_flagcx_wrapper",
        lambda: (
            lambda path: runtime.library,
            lambda ptr=None: ("buffer", ptr),
            object,
            runtime.dtype_enum,
            runtime.redop_enum,
            lambda: SimpleNamespace(internal=[0] * 128),
        ),
    )
    flagcx_module.dist = SimpleNamespace(
        get_rank=lambda group: 0,
        get_world_size=lambda group: 2,
        get_process_group_ranks=lambda group: [0, 1],
        broadcast=lambda tensor, src, group: None,
    )
    cls = flagcx_module.FlagCXCommunicator
    monkeypatch.setattr(cls, "_get_stream", lambda self: runtime.get_flagcx_stream())
    monkeypatch.setattr(
        cls,
        "_sync_current_stream",
        lambda self: runtime.events.append(("initial_warmup_sync",)),
    )

    comm = cls(group="cpu-group", device=runtime.device)

    assert comm.available and not comm.disabled
    assert comm._npu_comm_stream is runtime.communication
    assert runtime.stream_creations == 1
    names = [event[0] for event in runtime.events]
    assert names.index("init_comm") < names.index("create_stream")
    assert names.index("create_stream") < names.index("zeros")
    assert names.index("zeros") < names.index("allocate") < names.index("native")
    assert names[-1] == "initial_warmup_sync"
    assert runtime.native_calls == [
        runtime.expected_native_args(
            runtime.new_tensor("warmup", numel=1), runtime.outputs[0], "comm"
        )
    ]


def test_disabled_public_all_reduce_never_allocates_or_touches_npu(flagcx_module):
    runtime = _Runtime(flagcx_module)
    runtime.torch.npu = _ForbiddenNPU()
    comm = runtime.communicator()
    comm.disabled = True

    assert comm.all_reduce(runtime.new_tensor()) is None
    assert not runtime.events
    assert not runtime.outputs
