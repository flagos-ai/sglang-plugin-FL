import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_fl
from sglang_fl.distributed.communicator import CommunicatorFL


_GC_TARGET = "sglang.srt.distributed.parallel_state.GroupCoordinator"
_SCHEDULER_TARGET = "sglang.srt.managers.scheduler.Scheduler"


@pytest.fixture
def communicator_hooks(monkeypatch):
    registrations = {}

    class FakeHookRegistry:
        @staticmethod
        def register(target, hook, hook_type):
            registrations[target] = (hook, hook_type)

    class FakeHookType:
        AROUND = "around"

    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    plugins_mod = ModuleType("sglang.srt.plugins")
    hook_registry_mod = ModuleType("sglang.srt.plugins.hook_registry")
    hook_registry_mod.HookRegistry = FakeHookRegistry
    hook_registry_mod.HookType = FakeHookType

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.plugins", plugins_mod)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        hook_registry_mod,
    )

    sglang_fl._setup_communicator_hooks()
    return {target: hook for target, (hook, _) in registrations.items()}


def _communicator(flagcx=None):
    comm = CommunicatorFL.__new__(CommunicatorFL)
    comm.disabled = False
    comm.device = torch.device("cuda:0")
    comm._flagcx_comm = flagcx
    comm.all_reduce = Mock(return_value="fl-result")
    comm.reduce_scatter = Mock(return_value=None)
    comm.all_gather = Mock(return_value=None)
    comm.reduce_scatterv = Mock(return_value="fl-scatter-result")
    comm.all_gatherv = Mock(return_value="fl-gather-result")
    comm.send = Mock(return_value=None)
    comm.broadcast = Mock(return_value="fl-broadcast-result")
    return comm


@pytest.mark.parametrize(
    ("device_type", "available", "should_clear"),
    [
        ("musa", False, True),
        ("musa", True, False),
        ("musa", None, False),
        ("cuda", False, False),
        ("cpu", False, False),
    ],
)
def test_unavailable_pynccl_cannot_be_reenabled_by_musa_graph_capture(
    communicator_hooks, monkeypatch, device_type, available, should_clear
):
    hook = communicator_hooks[f"{_GC_TARGET}.__init__"]
    pynccl = SimpleNamespace(available=available, disabled=True)
    device_group = object()
    coordinator = SimpleNamespace()

    def initialize(self):
        self.device = SimpleNamespace(type=device_type)
        self.world_size = 2
        self.pynccl_comm = pynccl
        self.use_pynccl = True
        self.device_group = device_group
        self.cpu_group = object()
        self.rank_in_group = 0
        self.ranks = [0, 1]

    monkeypatch.setattr(
        "sglang_fl.distributed.communicator.CommunicatorFL",
        lambda **kwargs: SimpleNamespace(_flagcx_comm=None),
    )
    hook(initialize, coordinator)
    assert coordinator.device_group is device_group
    if should_clear:
        assert coordinator.pynccl_comm is None
        assert coordinator.use_pynccl is False
    else:
        assert coordinator.pynccl_comm is pynccl
        assert coordinator.use_pynccl is True


def test_nccl_without_flagcx_uses_original_sglang_all_reduce(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_reduce"]
    original = Mock(return_value="native-result")
    comm = _communicator(flagcx=None)
    coordinator = SimpleNamespace(fl_communicator=comm)
    tensor = object()

    assert hook(original, coordinator, tensor) == "native-result"
    original.assert_called_once_with(coordinator, tensor)
    comm.all_reduce.assert_not_called()


def test_active_flagcx_still_intercepts_all_reduce(communicator_hooks) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_reduce"]
    original = Mock(return_value="native-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    coordinator = SimpleNamespace(fl_communicator=comm)
    tensor = SimpleNamespace(device=torch.device("cuda:0"))

    assert hook(original, coordinator, tensor) == "fl-result"
    comm.all_reduce.assert_called_once_with(tensor)
    original.assert_not_called()


@pytest.mark.parametrize(
    ("outer_disabled", "flagcx_disabled"),
    [(True, False), (False, True)],
)
def test_disabled_communicator_uses_original_sglang_all_reduce(
    communicator_hooks,
    outer_disabled,
    flagcx_disabled,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_reduce"]
    original = Mock(return_value="native-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=flagcx_disabled))
    comm.disabled = outer_disabled
    coordinator = SimpleNamespace(fl_communicator=comm)
    tensor = object()

    assert hook(original, coordinator, tensor) == "native-result"
    original.assert_called_once_with(coordinator, tensor)
    comm.all_reduce.assert_not_called()


def test_legacy_vendor_communicator_without_active_marker_still_intercepts(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_reduce"]
    original = Mock(return_value="native-result")
    vendor_all_reduce = Mock(return_value="vendor-result")
    coordinator = SimpleNamespace(
        fl_communicator=SimpleNamespace(
            disabled=False,
            all_reduce=vendor_all_reduce,
        )
    )
    tensor = torch.empty(1)

    assert hook(original, coordinator, tensor) == "vendor-result"
    vendor_all_reduce.assert_called_once_with(tensor)
    original.assert_not_called()


@pytest.mark.parametrize(
    "tensor_device",
    [torch.device("cpu"), torch.device("cuda:1")],
)
def test_active_flagcx_uses_original_for_non_communicator_device(
    communicator_hooks, tensor_device
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_reduce"]
    original = Mock(return_value="native-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    coordinator = SimpleNamespace(fl_communicator=comm)
    tensor = SimpleNamespace(device=tensor_device)

    assert hook(original, coordinator, tensor) == "native-result"
    original.assert_called_once_with(coordinator, tensor)
    comm.all_reduce.assert_not_called()


def test_active_flagcx_tensor_hooks_preserve_cpu_native_paths(
    communicator_hooks,
) -> None:
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    coordinator = SimpleNamespace(
        fl_communicator=comm,
        rank_in_group=0,
        world_size=2,
    )
    cpu_tensor = torch.empty(1)
    output = torch.empty(1)
    cases = [
        ("all_reduce", (cpu_tensor,), {}, comm.all_reduce),
        ("all_gather", (cpu_tensor,), {}, comm.all_gather),
        ("_reduce_scatter_tensor", (output, cpu_tensor), {}, comm.reduce_scatter),
        ("_all_gather_into_tensor", (output, cpu_tensor), {}, comm.all_gather),
        ("reduce_scatterv", (cpu_tensor,), {}, comm.reduce_scatterv),
        ("all_gatherv", (cpu_tensor,), {}, comm.all_gatherv),
        ("send", (cpu_tensor,), {"dst": 1}, comm.send),
        ("broadcast", (cpu_tensor,), {"src": 1}, comm.broadcast),
    ]

    for method_name, args, kwargs, fl_method in cases:
        original = Mock(return_value="native-result")
        hook = communicator_hooks[f"{_GC_TARGET}.{method_name}"]

        assert hook(original, coordinator, *args, **kwargs) == "native-result"
        original.assert_called_once()
        fl_method.assert_not_called()


class _AcceleratorTensor(torch.Tensor):
    """Keep real CPU storage while exercising accelerator routing without NPU."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cpu(self):
        return False


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_npu_logits_all_gather_bypasses_native_shortcut(
    communicator_hooks, monkeypatch, dim, layout
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_gather"]
    values = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4)
    if layout == "transpose":
        values = values.transpose(0, 1)
    elif layout == "strided":
        values = values[::2]
    input_ = values.as_subclass(_AcceleratorTensor)
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    native = SimpleNamespace(
        disabled=False,
        all_gather=Mock(return_value="native-hccl-result"),
        quant_all_reduce=Mock(return_value="native-quant-result"),
    )
    coordinator = SimpleNamespace(
        fl_communicator=comm, npu_communicator=native, world_size=2
    )

    def original(self, tensor, dim=-1, output_tensor_list=None):
        # This upstream early return never reaches _all_gather_into_tensor,
        # which is the only gather entry point the old plugin intercepted.
        if not self.npu_communicator.disabled:
            return self.npu_communicator.all_gather(tensor, dim)
        raise AssertionError("The native NPU communicator must stay enabled")

    assert original(coordinator, input_, dim) == "native-hccl-result"
    native.all_gather.reset_mock()

    real_empty = torch.empty

    def empty_on_cpu(*args, **kwargs):
        assert kwargs["device"] == torch.device("cuda:0")
        return real_empty(*args, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr(torch, "empty", empty_on_cpu)

    def gather(output, tensor):
        assert tensor.is_contiguous(), "FlagCX reads a flat storage pointer"
        local = tensor.as_subclass(torch.Tensor)
        output.copy_(torch.cat([local, local + 100], dim=0))

    comm.all_gather.side_effect = gather
    result = hook(original, coordinator, input_, dim=dim)

    assert torch.equal(result, torch.cat([values, values + 100], dim=dim))
    comm.all_gather.assert_called_once()
    native.all_gather.assert_not_called()
    assert coordinator.npu_communicator is native
    assert native.disabled is False
    assert native.quant_all_reduce(input_) == "native-quant-result"


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "cross_device",
        "inactive",
        "outer_disabled",
        "flagcx_disabled",
        "single_rank",
        "output_list",
        "npu_missing",
        "npu_disabled",
        "hpu_active",
    ],
)
def test_npu_all_gather_preserves_other_routes(communicator_hooks, case) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_gather"]
    original = Mock(return_value="native-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    input_ = torch.arange(6).reshape(2, 3).as_subclass(_AcceleratorTensor)
    native = SimpleNamespace(disabled=False)
    coordinator = SimpleNamespace(
        fl_communicator=comm, npu_communicator=native, world_size=2
    )
    output_tensor_list = None
    dim = -1
    if case == "cpu":
        input_ = input_.as_subclass(torch.Tensor)
    elif case == "cross_device":
        comm.device = torch.device("cuda:1")
    elif case == "inactive":
        comm._flagcx_comm = None
    elif case == "outer_disabled":
        comm.disabled = True
    elif case == "flagcx_disabled":
        comm._flagcx_comm.disabled = True
    elif case == "single_rank":
        coordinator.world_size = 1
        input_ = torch.tensor(1).as_subclass(_AcceleratorTensor)
        dim = 100  # Upstream returns single-rank inputs before checking dim.
    elif case == "output_list":
        output_tensor_list = [object(), object()]
    elif case == "npu_missing":
        del coordinator.npu_communicator
    elif case == "npu_disabled":
        native.disabled = True
    elif case == "hpu_active":
        coordinator.hpu_communicator = SimpleNamespace(disabled=False)

    assert (
        hook(
            original,
            coordinator,
            input_,
            dim=dim,
            output_tensor_list=output_tensor_list,
        )
        == "native-result"
    )
    original.assert_called_once_with(
        coordinator, input_, dim=dim, output_tensor_list=output_tensor_list
    )
    comm.all_gather.assert_not_called()


def test_nccl_all_gatherv_forwards_preallocated_output_to_sglang(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_gatherv"]
    original = Mock(return_value="native-gather-result")
    comm = _communicator(flagcx=None)
    coordinator = SimpleNamespace(fl_communicator=comm)
    input_ = object()
    output = object()
    sizes = [2, 3]

    assert (
        hook(original, coordinator, input_, sizes=sizes, output=output)
        == "native-gather-result"
    )
    original.assert_called_once_with(coordinator, input_, sizes=sizes, output=output)
    comm.all_gatherv.assert_not_called()


def test_active_flagcx_all_gatherv_without_output_still_intercepts(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_gatherv"]
    original = Mock(return_value="native-gather-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    coordinator = SimpleNamespace(fl_communicator=comm)
    input_ = SimpleNamespace(device=torch.device("cuda:0"))
    sizes = [2, 3]

    assert hook(original, coordinator, input_, sizes=sizes) == "fl-gather-result"
    comm.all_gatherv.assert_called_once_with(input_, sizes=sizes)
    original.assert_not_called()


def test_active_flagcx_all_gatherv_with_output_uses_sglang_native_path(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_GC_TARGET}.all_gatherv"]
    original = Mock(return_value="native-gather-result")
    comm = _communicator(flagcx=SimpleNamespace(disabled=False))
    coordinator = SimpleNamespace(fl_communicator=comm)
    input_ = SimpleNamespace(device=torch.device("cuda:0"))
    output = object()
    sizes = [2, 3]

    assert (
        hook(original, coordinator, input_, sizes=sizes, output=output)
        == "native-gather-result"
    )
    original.assert_called_once_with(coordinator, input_, sizes=sizes, output=output)
    comm.all_gatherv.assert_not_called()


def test_pp_shutdown_waits_for_forwarded_request_then_exits(
    communicator_hooks,
) -> None:
    hook = communicator_hooks[f"{_SCHEDULER_TARGET}.get_next_batch_to_run"]
    original = Mock(return_value="plan")
    pending_work = [object()]
    commit_work = Mock(side_effect=lambda work: work.clear())
    scheduler = SimpleNamespace(
        gracefully_exit=True,
        ps=SimpleNamespace(pp_size=2),
        send_req_work=pending_work,
        _pp_commit_comm_work=commit_work,
    )

    with pytest.raises(SystemExit) as exc_info:
        hook(original, scheduler, running_batch=None, last_batch=None)

    assert exc_info.value.code == 0
    commit_work.assert_called_once_with(pending_work)
    assert pending_work == []
    original.assert_not_called()


@pytest.mark.parametrize(
    ("gracefully_exit", "pp_size"),
    [(False, 2), (True, 1)],
)
def test_pp_shutdown_hook_preserves_non_pp_scheduling(
    communicator_hooks, gracefully_exit, pp_size
) -> None:
    hook = communicator_hooks[f"{_SCHEDULER_TARGET}.get_next_batch_to_run"]
    original = Mock(return_value="plan")
    scheduler = SimpleNamespace(
        gracefully_exit=gracefully_exit,
        ps=SimpleNamespace(pp_size=pp_size),
        send_req_work=[],
        _pp_commit_comm_work=Mock(),
    )

    assert (
        hook(original, scheduler, running_batch="running", last_batch="last") == "plan"
    )
    original.assert_called_once_with(
        scheduler, running_batch="running", last_batch="last"
    )
    scheduler._pp_commit_comm_work.assert_not_called()
