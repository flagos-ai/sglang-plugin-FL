# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for SGLang-FL CommunicatorFL.

These tests use mocks to validate core initialization, backend routing, and
global-rank mapping without starting a real torch.distributed process group.
Detailed collective and tensor-dict behavior is covered by functional tests.
"""

from __future__ import annotations

from unittest.mock import Mock

import torch


class _FakePlatform:
    def __init__(self, backend: str) -> None:
        self._dist_backend = backend


class _FakeFlagCX:
    disabled = False
    available = True

    def __init__(self) -> None:
        self.calls = []

    def all_reduce(self, tensor):
        self.calls.append(("all_reduce", tuple(tensor.shape)))
        return tensor + 1

    def send(self, tensor, dst):
        self.calls.append(("send", dst, tuple(tensor.shape)))

    def recv(self, tensor, src):
        self.calls.append(("recv", src, tuple(tensor.shape)))


def _make_comm(*, flagcx=None, world_size: int = 2, rank: int = 0):
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
    assert TensorMetadata("cpu", torch.float32, torch.Size([2])).size == torch.Size([2])


def test_world_size_one_does_not_create_flagcx(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod
    import sglang_fl.platform as platform_mod
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    fake_ctor = Mock(return_value=_FakeFlagCX())
    monkeypatch.setattr(platform_mod, "PlatformFL", lambda: _FakePlatform("flagcx"))
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


def test_flagcx_creation_failure_falls_back_to_torch(monkeypatch) -> None:
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


def test_all_reduce_uses_flagcx_and_copies_back(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "all_reduce", Mock())
    flagcx = _FakeFlagCX()
    comm = _make_comm(flagcx=flagcx)
    tensor = torch.tensor([1.0, 2.0])

    assert comm.all_reduce(tensor) is tensor
    assert torch.equal(tensor, torch.tensor([2.0, 3.0]))
    assert flagcx.calls == [("all_reduce", (2,))]
    comm_mod.dist.all_reduce.assert_not_called()


def test_all_reduce_without_flagcx_uses_dist(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    dist_all_reduce = Mock()
    monkeypatch.setattr(comm_mod.dist, "all_reduce", dist_all_reduce)
    comm = _make_comm()
    tensor = torch.ones(2)

    assert comm.all_reduce(tensor) is tensor
    dist_all_reduce.assert_called_once_with(tensor, group="device-group")


def test_send_recv_fallback_use_dist_global_ranks(monkeypatch) -> None:
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