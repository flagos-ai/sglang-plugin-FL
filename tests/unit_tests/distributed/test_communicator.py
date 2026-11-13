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
    comm._pp_aux_comm = None
    return comm


class _FakeAuxComm:
    available = True

    def __init__(self) -> None:
        self.calls = []

    def all_gather(self, output, input_):
        self.calls.append(("all_gather", tuple(output.shape), tuple(input_.shape)))


def _make_all_gather_group(fl_communicator):
    group = Mock()
    group.world_size = 2
    group.rank_in_group = 0
    group.fl_communicator = fl_communicator
    return group


def _recv_proxy_metadata(src):
    from sglang_fl.distributed.communicator import TensorMetadata

    return [
        ("hidden_states", TensorMetadata("cpu", torch.float32, torch.Size([4]))),
        ("__msg_type__", "proxy"),
    ]


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


def test_pp_aux_comm_without_flagcx_is_none(monkeypatch) -> None:
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    fake_ctor = Mock()
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", fake_ctor)
    comm = _make_comm()

    assert comm.pp_aux_comm() is None
    fake_ctor.assert_not_called()


def test_pp_aux_comm_is_separate_and_created_once(monkeypatch) -> None:
    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    aux = _FakeAuxComm()
    fake_ctor = Mock(return_value=aux)
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", fake_ctor)
    comm = _make_comm(flagcx=_FakeFlagCX())

    assert comm.pp_aux_comm() is aux
    assert comm.pp_aux_comm() is aux
    assert aux is not comm._flagcx_comm
    fake_ctor.assert_called_once_with(group="cpu-group", device=torch.device("cpu"))


def test_pp_aux_comm_init_failure_raises(monkeypatch) -> None:
    import pytest

    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    aux = _FakeAuxComm()
    aux.available = False
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", Mock(return_value=aux))
    comm = _make_comm(flagcx=_FakeFlagCX())

    with pytest.raises(RuntimeError):
        comm.init_pp_aux_comm()


def test_pp_aux_comm_after_init_does_not_warn(monkeypatch, caplog) -> None:
    import logging

    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    aux = _FakeAuxComm()
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", Mock(return_value=aux))
    comm = _make_comm(flagcx=_FakeFlagCX())

    comm.init_pp_aux_comm()
    with caplog.at_level(logging.WARNING, logger="sglang_fl.distributed.communicator"):
        assert comm.pp_aux_comm() is aux
    assert "first PP receive" not in caplog.text


def test_pp_aux_comm_created_on_first_use_warns(monkeypatch, caplog) -> None:
    import logging

    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    aux = _FakeAuxComm()
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", Mock(return_value=aux))
    comm = _make_comm(flagcx=_FakeFlagCX())

    with caplog.at_level(logging.WARNING, logger="sglang_fl.distributed.communicator"):
        assert comm.pp_aux_comm() is aux
    assert "first PP receive" in caplog.text


def test_pp_aux_comm_not_created_during_graph_capture(monkeypatch) -> None:
    import pytest

    import sglang_fl.distributed.device_communicators.flagcx as flagcx_mod

    fake_ctor = Mock(return_value=_FakeAuxComm())
    monkeypatch.setattr(flagcx_mod, "FlagCXCommunicator", fake_ctor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    comm = _make_comm(flagcx=_FakeFlagCX())

    with pytest.raises(RuntimeError):
        comm.pp_aux_comm()
    fake_ctor.assert_not_called()


def test_recv_tensor_dict_all_gather_uses_pp_aux_comm(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "recv", Mock())
    tp_comm = _make_comm(flagcx=_FakeFlagCX())
    tp_comm._pp_aux_comm = _FakeAuxComm()
    group = _make_all_gather_group(tp_comm)
    pp_comm = _make_comm(flagcx=_FakeFlagCX(), world_size=2, rank=1)

    out = pp_comm.recv_tensor_dict(
        src=0, recv_object_fn=_recv_proxy_metadata, all_gather_group=group
    )

    assert tp_comm._pp_aux_comm.calls == [("all_gather", (4,), (2,))]
    group._all_gather_into_tensor.assert_not_called()
    assert out["hidden_states"].shape == torch.Size([4])
    assert out["__msg_type__"] == "proxy"


def test_recv_tensor_dict_without_flagcx_uses_group_all_gather(monkeypatch) -> None:
    from sglang_fl.distributed import communicator as comm_mod

    monkeypatch.setattr(comm_mod.dist, "irecv", Mock())
    group = _make_all_gather_group(_make_comm())
    pp_comm = _make_comm(world_size=2, rank=1)

    pp_comm.recv_tensor_dict(
        src=0, recv_object_fn=_recv_proxy_metadata, all_gather_group=group
    )

    group._all_gather_into_tensor.assert_called_once()


def test_create_pp_aux_comm_uses_attention_tp_group(monkeypatch) -> None:
    from sglang.srt.distributed import parallel_state

    from sglang_fl.distributed import communicator as comm_mod

    tp_comm = Mock()
    monkeypatch.setattr(
        parallel_state, "get_attn_tp_group", lambda: _make_all_gather_group(tp_comm)
    )

    comm_mod.create_pp_aux_comm()

    tp_comm.init_pp_aux_comm.assert_called_once_with()


def test_create_pp_aux_comm_warns_without_attention_tp_group(
    monkeypatch, caplog
) -> None:
    import logging

    from sglang.srt.distributed import parallel_state

    from sglang_fl.distributed import communicator as comm_mod

    def _missing():
        raise AssertionError("attention tensor parallel group is not initialized")

    monkeypatch.setattr(parallel_state, "get_attn_tp_group", _missing)

    with caplog.at_level(logging.WARNING, logger="sglang_fl.distributed.communicator"):
        comm_mod.create_pp_aux_comm()
    assert "created on first use" in caplog.text
