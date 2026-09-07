import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_fl
from sglang_fl.distributed.communicator import CommunicatorFL


_GC_TARGET = "sglang.srt.distributed.parallel_state.GroupCoordinator"


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
    original.assert_called_once_with(
        coordinator, input_, sizes=sizes, output=output
    )
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
    original.assert_called_once_with(
        coordinator, input_, sizes=sizes, output=output
    )
    comm.all_gatherv.assert_not_called()
