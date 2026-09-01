# Copyright (c) 2026 BAAI. All rights reserved.

"""Functional smoke tests for SGLang-FL distributed collectives.

These tests use lightweight mocks instead of initializing a real distributed
process group. The goal is to validate importability, routing decisions, and
basic tensor/metadata semantics in ``CommunicatorFL``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.functional]


class _FakePlatform:
    def __init__(self, backend: str = "nccl") -> None:
        self._dist_backend = backend


class _FakeFlagCX:
    disabled = False
    available = True

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(("all_reduce", tensor.clone()))
        return tensor + 1

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        self.calls.append(("reduce_scatter", tuple(output.shape), tuple(input_.shape)))
        output.copy_(input_[: output.shape[0]])

    def reduce_scatterv(self, output: torch.Tensor, input_: torch.Tensor, sizes) -> None:
        self.calls.append(("reduce_scatterv", list(sizes)))
        output.copy_(input_[: output.shape[0]])

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        self.calls.append(("all_gather", tuple(output.shape), tuple(input_.shape)))
        output[: input_.shape[0]].copy_(input_)

    def all_gatherv(self, output: torch.Tensor, input_: torch.Tensor, sizes) -> None:
        self.calls.append(("all_gatherv", list(sizes)))
        output[: input_.shape[0]].copy_(input_)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.calls.append(("broadcast", src, tuple(tensor.shape)))

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self.calls.append(("send", dst, tuple(tensor.shape)))

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        self.calls.append(("recv", src, tuple(tensor.shape)))
        tensor.fill_(src)

    def group_start(self) -> None:
        self.calls.append(("group_start",))

    def group_end(self) -> None:
        self.calls.append(("group_end",))


class _FakeWork:
    def __init__(self) -> None:
        self.wait = Mock()


def _make_comm(*, world_size: int = 2, rank: int = 0, flagcx=None):
    from sglang_fl.distributed.communicator import CommunicatorFL

    comm = CommunicatorFL.__new__(CommunicatorFL)
    comm.cpu_group = "cpu-group"
    comm.device = torch.device("cpu")
    comm.device_group = "device-group"
    comm.world_size = world_size
    comm.rank_in_group = rank
    comm.ranks = list(range(10, 10 + world_size))
    comm._dist_backend = "flagcx" if flagcx else "nccl"
    comm._flagcx_comm = flagcx
    return comm


def test_communicator_fl_import() -> None:
    from sglang_fl.distributed.communicator import CommunicatorFL, TensorMetadata

    assert CommunicatorFL.__name__ == "CommunicatorFL"
    assert TensorMetadata("cpu", torch.float32, torch.Size([1])).device == "cpu"


def test_flagcx_communicator_import() -> None:
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    assert FlagCXCommunicator.__name__ == "FlagCXCommunicator"


def test_world_size_one_skips_flagcx_creation(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    import sglang_fl.platform as platform_mod
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    monkeypatch.setattr(platform_mod, "PlatformFL", lambda: _FakePlatform("flagcx"))
    fake_ctor = Mock(return_value=_FakeFlagCX())
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", fake_ctor)

    comm = comm_mod.CommunicatorFL(
        cpu_group="cpu-group",
        device=torch.device("cpu"),
        device_group="device-group",
        world_size=1,
        rank_in_group=0,
        ranks=[0],
    )

    assert comm._dist_backend == "flagcx"
    assert comm._flagcx_comm is None
    fake_ctor.assert_not_called()


def test_flagcx_creation_failure_falls_back(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    import sglang_fl.platform as platform_mod
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    monkeypatch.setattr(platform_mod, "PlatformFL", lambda: _FakePlatform("flagcx"))
    monkeypatch.setattr(
        flagcx_mod,
        "FlagCXCommunicator",
        Mock(side_effect=RuntimeError("flagcx unavailable")),
    )

    comm = comm_mod.CommunicatorFL(
        cpu_group="cpu-group",
        device=torch.device("cpu"),
        device_group="device-group",
        world_size=2,
        rank_in_group=0,
        ranks=[0, 1],
    )

    assert comm._dist_backend == "flagcx"
    assert comm._flagcx_comm is None


def test_flagcx_backend_creates_communicator(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    import sglang_fl.platform as platform_mod
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    fake_flagcx = _FakeFlagCX()
    fake_ctor = Mock(return_value=fake_flagcx)
    monkeypatch.setattr(platform_mod, "PlatformFL", lambda: _FakePlatform("flagcx"))
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", fake_ctor)

    comm = comm_mod.CommunicatorFL(
        cpu_group="cpu-group",
        device=torch.device("cpu"),
        device_group="device-group",
        world_size=2,
        rank_in_group=0,
        ranks=[0, 1],
    )

    assert comm._flagcx_comm is fake_flagcx
    fake_ctor.assert_called_once_with(group="cpu-group", device=torch.device("cpu"))


def test_all_reduce_routes_to_flagcx_and_copies_back(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "all_reduce", Mock())
    flagcx = _FakeFlagCX()
    comm = _make_comm(flagcx=flagcx)
    tensor = torch.tensor([1.0, 2.0])

    out = comm.all_reduce(tensor)

    assert out is tensor
    assert torch.equal(tensor, torch.tensor([2.0, 3.0]))
    assert flagcx.calls[0][0] == "all_reduce"
    comm_mod.dist.all_reduce.assert_not_called()


def test_all_reduce_falls_back_to_torch_when_flagcx_disabled(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    flagcx = _FakeFlagCX()
    flagcx.disabled = True
    dist_all_reduce = Mock()
    monkeypatch.setattr(comm_mod.dist, "all_reduce", dist_all_reduce)
    comm = _make_comm(flagcx=flagcx)
    tensor = torch.tensor([1.0])

    assert comm.all_reduce(tensor) is tensor
    dist_all_reduce.assert_called_once_with(tensor, group="device-group")
    assert flagcx.calls == []


def test_reduce_scatter_routes_to_flagcx(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "reduce_scatter_tensor", Mock())
    flagcx = _FakeFlagCX()
    comm = _make_comm(flagcx=flagcx)
    input_ = torch.arange(4.0)
    output = torch.empty(2)

    comm.reduce_scatter(output, input_)

    assert torch.equal(output, torch.tensor([0.0, 1.0]))
    assert flagcx.calls[0][0] == "reduce_scatter"
    comm_mod.dist.reduce_scatter_tensor.assert_not_called()


def test_reduce_scatter_falls_back_to_torch(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_reduce_scatter = Mock()
    monkeypatch.setattr(comm_mod.dist, "reduce_scatter_tensor", dist_reduce_scatter)
    comm = _make_comm()
    input_ = torch.arange(4.0)
    output = torch.empty(2)

    comm.reduce_scatter(output, input_)

    dist_reduce_scatter.assert_called_once_with(output, input_, group="device-group")


def test_all_gather_routes_to_flagcx(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "all_gather_into_tensor", Mock())
    flagcx = _FakeFlagCX()
    comm = _make_comm(flagcx=flagcx)
    input_ = torch.tensor([3.0, 4.0])
    output = torch.empty(4)

    comm.all_gather(output, input_)

    assert torch.equal(output[:2], input_)
    assert flagcx.calls[0][0] == "all_gather"
    comm_mod.dist.all_gather_into_tensor.assert_not_called()


def test_all_gather_falls_back_to_torch(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_all_gather = Mock()
    monkeypatch.setattr(comm_mod.dist, "all_gather_into_tensor", dist_all_gather)
    comm = _make_comm()
    input_ = torch.tensor([3.0, 4.0])
    output = torch.empty(4)

    comm.all_gather(output, input_)

    dist_all_gather.assert_called_once_with(output, input_, group="device-group")


def test_send_recv_route_to_flagcx(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "send", Mock())
    monkeypatch.setattr(comm_mod.dist, "recv", Mock())
    flagcx = _FakeFlagCX()
    comm = _make_comm(flagcx=flagcx)
    send_tensor = torch.ones(2)
    recv_tensor = torch.empty(2)

    comm.send(send_tensor, dst=1)
    comm.recv(recv_tensor, src=1)

    assert ("send", 1, (2,)) in flagcx.calls
    assert ("recv", 1, (2,)) in flagcx.calls
    assert torch.equal(recv_tensor, torch.tensor([1.0, 1.0]))
    comm_mod.dist.send.assert_not_called()
    comm_mod.dist.recv.assert_not_called()


def test_send_recv_fallback_uses_global_ranks(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_send = Mock()
    dist_recv = Mock()
    monkeypatch.setattr(comm_mod.dist, "send", dist_send)
    monkeypatch.setattr(comm_mod.dist, "recv", dist_recv)
    comm = _make_comm(world_size=2, rank=0)
    tensor = torch.ones(2)

    comm.send(tensor, dst=1)
    comm.recv(tensor, src=1)

    dist_send.assert_called_once_with(tensor, 11, "device-group")
    dist_recv.assert_called_once_with(tensor, 11, "device-group")


def test_broadcast_fallback_uses_global_src_rank(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_broadcast = Mock()
    monkeypatch.setattr(comm_mod.dist, "broadcast", dist_broadcast)
    comm = _make_comm(world_size=2, rank=0)
    tensor = torch.ones(2)

    assert comm.broadcast(tensor, src=1) is tensor
    dist_broadcast.assert_called_once_with(tensor, src=11, group="device-group")


def test_reduce_scatterv_sizes_select_rank_chunk(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(
        comm_mod.dist,
        "reduce_scatter_tensor",
        lambda output, input_, group=None: output.copy_(input_[: output.shape[0]]),
    )
    input_ = torch.arange(10.0).reshape(5, 2)

    rank0 = _make_comm(world_size=2, rank=0).reduce_scatterv(input_, sizes=[2, 3])
    rank1 = _make_comm(world_size=2, rank=1).reduce_scatterv(input_, sizes=[2, 3])

    assert rank0.shape == (2, 2)
    assert rank1.shape == (3, 2)


def test_reduce_scatterv_invalid_sizes_asserts() -> None:
    comm = _make_comm(world_size=2, rank=0)
    with pytest.raises(AssertionError):
        comm.reduce_scatterv(torch.arange(8.0).reshape(4, 2), sizes=[2, 3])


def test_all_gatherv_output_shape_and_torch_path(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    calls = []

    def fake_all_gather(output, input_, group=None):
        calls.append((tuple(output.shape), tuple(input_.shape), group))
        output[: input_.shape[0]].copy_(input_)

    monkeypatch.setattr(comm_mod.dist, "all_gather_into_tensor", fake_all_gather)
    comm = _make_comm(world_size=2, rank=1)
    input_ = torch.ones(3, 2)

    outputs = comm.all_gatherv(input_, sizes=[2, 3])

    assert isinstance(outputs, list)
    assert outputs[0].shape == (5, 2)
    assert calls == [((5, 2), (3, 2), "device-group")]


def test_all_gatherv_equal_sizes_uses_regular_all_gather(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_all_gather = Mock(side_effect=lambda output, input_, group=None: output.zero_())
    monkeypatch.setattr(comm_mod.dist, "all_gather_into_tensor", dist_all_gather)
    comm = _make_comm(world_size=2, rank=0)

    outputs = comm.all_gatherv(torch.ones(2, 2), sizes=[2, 2])

    assert outputs[0].shape == (4, 2)
    dist_all_gather.assert_called_once()


def test_broadcast_tensor_dict_sender_metadata_and_tensor_route(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    from sglang_fl.distributed.communicator import TensorMetadata

    dist_broadcast = Mock()
    monkeypatch.setattr(comm_mod.dist, "broadcast", dist_broadcast)
    comm = _make_comm(world_size=2, rank=0)
    tensor_dict = {
        "tensor": torch.ones(2),
        "empty": torch.empty(0),
        "value": "metadata-only",
    }
    sent_objects = []

    def broadcast_object_fn(obj, src):
        sent_objects.append((obj, src))
        return obj

    out = comm.broadcast_tensor_dict(tensor_dict, src=0, rank_in_group=0, broadcast_object_fn=broadcast_object_fn)

    assert out is tensor_dict
    metadata = sent_objects[0][0]
    assert metadata[0][0] == "tensor"
    assert isinstance(metadata[0][1], TensorMetadata)
    assert metadata[1][0] == "empty"
    assert isinstance(metadata[1][1], TensorMetadata)
    assert metadata[2] == ("value", "metadata-only")
    dist_broadcast.assert_called_once_with(tensor_dict["tensor"], src=10, group="cpu-group")


def test_broadcast_tensor_dict_receiver_allocates_from_metadata(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    from sglang_fl.distributed.communicator import TensorMetadata

    dist_broadcast = Mock()
    monkeypatch.setattr(comm_mod.dist, "broadcast", dist_broadcast)
    comm = _make_comm(world_size=2, rank=1)
    metadata = [
        ("tensor", TensorMetadata("cpu", torch.float32, torch.Size([2]))),
        ("empty", TensorMetadata("cpu", torch.float32, torch.Size([0]))),
        ("value", 123),
    ]

    out = comm.broadcast_tensor_dict(None, src=0, rank_in_group=1, broadcast_object_fn=lambda obj, src: metadata)

    assert out["tensor"].shape == (2,)
    assert out["empty"].shape == (0,)
    assert out["value"] == 123
    dist_broadcast.assert_called_once()


def test_send_tensor_dict_metadata_async_and_skips_empty(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    from sglang_fl.distributed.communicator import TensorMetadata

    dist_send = Mock()
    monkeypatch.setattr(comm_mod.dist, "send", dist_send)
    comm = _make_comm(world_size=2, rank=0)
    works = [object()]
    sent = []

    def send_object_fn(metadata, dst, async_send=False):
        sent.append((metadata, dst, async_send))
        return works

    result = comm.send_tensor_dict(
        {"tensor": torch.ones(2), "empty": torch.empty(0), "value": "x"},
        dst=1,
        send_object_fn=send_object_fn,
    )

    assert result is works
    metadata = sent[0][0]
    assert sent[0][1:] == (1, True)
    assert isinstance(metadata[0][1], TensorMetadata)
    assert isinstance(metadata[1][1], TensorMetadata)
    assert metadata[2] == ("value", "x")
    dist_send.assert_called_once()


def test_recv_tensor_dict_allocates_and_waits(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    from sglang_fl.distributed.communicator import TensorMetadata

    work = _FakeWork()
    irecv = Mock(return_value=work)
    monkeypatch.setattr(comm_mod.dist, "irecv", irecv)
    comm = _make_comm(world_size=2, rank=1)
    metadata = [
        ("tensor", TensorMetadata("cpu", torch.float32, torch.Size([2]))),
        ("empty", TensorMetadata("cpu", torch.float32, torch.Size([0]))),
        ("value", "x"),
    ]

    out = comm.recv_tensor_dict(src=0, recv_object_fn=lambda src: metadata)

    assert out["tensor"].shape == (2,)
    assert out["empty"].shape == (0,)
    assert out["value"] == "x"
    irecv.assert_called_once()
    work.wait.assert_called_once()


def test_recv_tensor_dict_all_gather_restores_full_tensor(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    from sglang_fl.distributed.communicator import TensorMetadata

    monkeypatch.setattr(comm_mod.dist, "irecv", Mock(return_value=_FakeWork()))
    comm = _make_comm(world_size=2, rank=1)
    metadata = [("tensor", TensorMetadata("cpu", torch.float32, torch.Size([4])))]
    all_gather_group = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        _all_gather_into_tensor=Mock(side_effect=lambda full, part: full.fill_(7)),
    )

    out = comm.recv_tensor_dict(
        src=0,
        recv_object_fn=lambda src: metadata,
        all_gather_group=all_gather_group,
    )

    assert torch.equal(out["tensor"], torch.full((4,), 7.0))
    all_gather_group._all_gather_into_tensor.assert_called_once()
