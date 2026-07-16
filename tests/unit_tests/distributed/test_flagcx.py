# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the FlagCX device communicator wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


class _FakeFlagCXLibrary:
    def __init__(self, library_path: str) -> None:
        self.library_path = library_path


class _FakeFlagCXId:
    def __init__(self) -> None:
        self.internal = [0] * 128


def _fake_wrapper_tuple():
    return (
        _FakeFlagCXLibrary,
        lambda ptr=None: ptr,
        object,
        object,
        object,
        _FakeFlagCXId,
    )


def _disabled_comm():
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm.disabled = True
    comm.available = False
    comm.device = torch.device("cpu")
    return comm


def test_import_flagcx_wrapper_requires_flagcx_path(monkeypatch) -> None:
    from sglang_fl.distributed.device_communicators.flagcx import _import_flagcx_wrapper

    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    with pytest.raises(RuntimeError, match="FLAGCX_PATH environment variable is not set"):
        _import_flagcx_wrapper()


def test_import_flagcx_wrapper_rejects_invalid_path(monkeypatch) -> None:
    from sglang_fl.distributed.device_communicators.flagcx import _import_flagcx_wrapper

    monkeypatch.setenv("FLAGCX_PATH", "/path/that/does/not/exist")

    with pytest.raises(RuntimeError, match="is not a valid directory"):
        _import_flagcx_wrapper()


def test_world_size_one_disables_communicator(monkeypatch, tmp_path: Path) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(flagcx_mod, "_import_flagcx_wrapper", lambda: _fake_wrapper_tuple())
    monkeypatch.setattr(flagcx_mod.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(flagcx_mod.dist, "get_world_size", Mock(return_value=1))

    comm = flagcx_mod.FlagCXCommunicator(group="group", device=torch.device("cpu"))

    assert comm.disabled is True
    assert comm.available is False
    assert comm.group == "group"


def test_destroy_marks_communicator_unavailable() -> None:
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm.comm = object()
    comm.available = True
    comm.disabled = False
    comm.flagcx = Mock()

    comm.destroy()

    assert comm.comm is None
    assert comm.available is False
    assert comm.disabled is True
    comm.flagcx.flagcxCommDestroy.assert_called_once()


def test_disabled_all_reduce_returns_early() -> None:
    comm = _disabled_comm()

    assert comm.all_reduce(torch.ones(2)) is None


def test_disabled_reduce_scatter_returns_early() -> None:
    comm = _disabled_comm()

    assert comm.reduce_scatter(torch.empty(2), torch.ones(4)) is None


def test_disabled_all_gather_returns_early() -> None:
    comm = _disabled_comm()

    assert comm.all_gather(torch.empty(4), torch.ones(2)) is None


def test_disabled_reduce_scatterv_returns_early() -> None:
    comm = _disabled_comm()

    assert comm.reduce_scatterv(torch.empty(2), torch.ones(4), sizes=[2, 2]) is None


def test_disabled_all_gatherv_returns_early() -> None:
    comm = _disabled_comm()

    assert comm.all_gatherv(torch.empty(4), torch.ones(2), sizes=[2, 2]) is None


def test_disabled_send_recv_broadcast_return_early() -> None:
    comm = _disabled_comm()
    tensor = torch.ones(2)

    assert comm.broadcast(tensor, src=0) is None
    assert comm.send(tensor, dst=1) is None
    assert comm.recv(tensor, src=0) is None
