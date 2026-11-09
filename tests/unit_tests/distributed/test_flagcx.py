# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the FlagCX device communicator wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch


@pytest.fixture(autouse=True)
def clear_flagcx_api_cache():
    from sglang_fl.distributed.device_communicators.flagcx import _import_flagcx_api

    _import_flagcx_api.cache_clear()
    try:
        yield
    finally:
        _import_flagcx_api.cache_clear()

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

    with pytest.raises(
        RuntimeError, match="FLAGCX_PATH environment variable is not set"
    ):
        _import_flagcx_wrapper()


def test_import_flagcx_wrapper_rejects_invalid_path(monkeypatch) -> None:
    from sglang_fl.distributed.device_communicators.flagcx import _import_flagcx_wrapper

    monkeypatch.setenv("FLAGCX_PATH", "/path/that/does/not/exist")

    with pytest.raises(RuntimeError, match="is not a valid directory"):
        _import_flagcx_wrapper()


def test_world_size_one_disables_communicator(monkeypatch, tmp_path: Path) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(
        flagcx_mod, "_import_flagcx_wrapper", lambda: _fake_wrapper_tuple()
    )
    monkeypatch.setattr(flagcx_mod.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(flagcx_mod.dist, "get_world_size", Mock(return_value=1))

    comm = flagcx_mod.FlagCXCommunicator(group="group", device=torch.device("cpu"))

    assert comm.disabled is True
    assert comm.available is False
    assert comm.group == "group"


def test_del_destroys_communicator() -> None:
    # FlagCXCommunicator.destroy() was removed in #32 and replaced by __del__
    # (GC-based teardown). Test the actual teardown path: __del__ calls
    # flagcxCommDestroy and nulls the comm handle. Unlike the old destroy(),
    # __del__ does not flip available/disabled, so those are not asserted.
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm.comm = object()
    comm.flagcx = Mock()
    comm._stream_cache = {}

    comm.__del__()

    assert comm.comm is None
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


def test_all_gather_waits_for_flagcx_completion() -> None:
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    events = Mock()
    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.comm = object()
    comm.flagcx = SimpleNamespace(flagcxAllGather=lambda *args: events("all_gather"))
    comm._buffer_type = lambda ptr: ptr
    comm._dtype_enum = SimpleNamespace(from_torch=lambda dtype: dtype)
    comm._get_stream = lambda: object()
    comm._is_current_stream_capturing = lambda: False
    comm._sync_current_stream = lambda: events("synchronize")

    comm.all_gather(torch.empty(4), torch.ones(2))

    assert events.call_args_list == [call("all_gather"), call("synchronize")]


def test_all_gather_does_not_synchronize_during_graph_capture() -> None:
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    events = Mock()
    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.comm = object()
    comm.flagcx = SimpleNamespace(flagcxAllGather=lambda *args: events("all_gather"))
    comm._buffer_type = lambda ptr: ptr
    comm._dtype_enum = SimpleNamespace(from_torch=lambda dtype: dtype)
    comm._get_stream = lambda: object()
    comm._is_current_stream_capturing = lambda: True
    comm._sync_current_stream = lambda: events("synchronize")

    comm.all_gather(torch.empty(4), torch.ones(2))

    assert events.call_args_list == [call("all_gather")]


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


def _mock_flagcx_api_discovery(
    monkeypatch, flagcx_mod, *, package_exists=True, api_exists=True
):
    spec = SimpleNamespace(submodule_search_locations=["/unused/flagcx"])
    monkeypatch.setattr(
        flagcx_mod, "find_spec", Mock(return_value=spec if package_exists else None)
    )
    monkeypatch.setattr(
        flagcx_mod,
        "PathFinder",
        SimpleNamespace(find_spec=Mock(return_value=object() if api_exists else None)),
    )


@pytest.mark.parametrize("missing_new", [None, "flagcx", "flagcx.api"])
def test_flagcx_api_selection_and_stream_cache(monkeypatch, tmp_path, missing_new):
    import ctypes
    import sys

    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    class Stream(ctypes.c_void_p):
        pass

    symbols = _fake_wrapper_tuple()
    api = SimpleNamespace(
        **dict(
            zip(
                (
                    "FLAGCXLibrary",
                    "buffer_type",
                    "flagcxComm_t",
                    "flagcxDataTypeEnum",
                    "flagcxRedOpTypeEnum",
                    "flagcxUniqueId",
                ),
                symbols,
            )
        ),
        flagcxStream_t=Stream,
    )

    def import_api(name):
        assert name == (
            "flagcx.api"
            if missing_new is None
            else "plugin.interservice.flagcx_wrapper"
        )
        return api

    _mock_flagcx_api_discovery(
        monkeypatch,
        flagcx_mod,
        package_exists=missing_new != "flagcx",
        api_exists=missing_new is None,
    )
    importer = Mock(side_effect=import_api)
    monkeypatch.setattr(flagcx_mod, "import_module", importer)
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert flagcx_mod._import_flagcx_wrapper() == symbols

    def copy_stream(out, handle):
        assert handle.value in (42, 43)
        out._obj.value = 123
        return 0

    copy = Mock(side_effect=copy_stream)
    comm = flagcx_mod.FlagCXCommunicator.__new__(flagcx_mod.FlagCXCommunicator)
    comm.comm = None
    comm._stream_cache = {}
    comm._get_raw_stream_handle = lambda: 42
    comm.flagcx = SimpleNamespace(
        devHandle=SimpleNamespace(contents=SimpleNamespace(streamCopy=copy)),
        FLAGCX_CHECK=Mock(),
    )
    stream = comm._get_stream()
    assert isinstance(stream, Stream)
    assert stream.value == 123
    assert comm._get_stream() is stream
    copy.assert_called_once()
    comm.flagcx.FLAGCX_CHECK.assert_called_once_with(0)
    selected = (
        "flagcx.api" if missing_new is None else "plugin.interservice.flagcx_wrapper"
    )
    comm._get_raw_stream_handle = lambda: 43
    other_stream = comm._get_stream()
    assert isinstance(other_stream, Stream)
    assert other_stream is not stream
    assert comm._get_stream() is other_stream
    assert copy.call_count == 2
    assert importer.call_args_list == [call(selected)]
    flagcx_mod.find_spec.assert_called_once_with("flagcx")
    comm._stream_cache.clear()


@pytest.mark.parametrize("dependency", ["vendor_dependency", "flagcx._C"])
def test_flagcx_api_dependency_error_does_not_fall_back(monkeypatch, dependency):
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    _mock_flagcx_api_discovery(monkeypatch, flagcx_mod)

    error = ModuleNotFoundError("missing internal dependency", name=dependency)
    importer = Mock(side_effect=error)
    monkeypatch.setattr(flagcx_mod, "import_module", importer)
    with pytest.raises(ModuleNotFoundError) as exc:
        flagcx_mod._import_flagcx_api()
    assert exc.value is error
    importer.assert_called_once_with("flagcx.api")


def test_flagcx_api_import_error_does_not_fall_back(monkeypatch):
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    _mock_flagcx_api_discovery(monkeypatch, flagcx_mod)

    error = ImportError("shared library load failed")
    importer = Mock(side_effect=error)
    monkeypatch.setattr(flagcx_mod, "import_module", importer)
    with pytest.raises(ImportError) as exc:
        flagcx_mod._import_flagcx_api()
    assert exc.value is error
    importer.assert_called_once_with("flagcx.api")


def test_flagcx_api_reports_missing_legacy_module(monkeypatch):
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    _mock_flagcx_api_discovery(monkeypatch, flagcx_mod, package_exists=False)
    missing_old = ModuleNotFoundError("missing legacy API", name="plugin")
    importer = Mock(side_effect=missing_old)
    monkeypatch.setattr(flagcx_mod, "import_module", importer)
    with pytest.raises(ModuleNotFoundError) as exc:
        flagcx_mod._import_flagcx_api()
    assert exc.value is missing_old
    importer.assert_called_once_with("plugin.interservice.flagcx_wrapper")


@pytest.mark.parametrize("has_api", [False, True])
def test_flagcx_api_discovery_avoids_legacy_package_initialization(
    monkeypatch, tmp_path, has_api
):
    import importlib
    import sys

    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    package = tmp_path / "flagcx"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise ImportError("libcudart.so.12: cannot open shared object file")\n'
    )
    if has_api:
        (package / "api.py").write_text("")
    for name in list(sys.modules):
        if name == "flagcx" or name.startswith("flagcx."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    legacy = object()

    def import_api(name):
        if name == "plugin.interservice.flagcx_wrapper":
            return legacy
        return importlib.import_module(name)

    importer = Mock(side_effect=import_api)
    monkeypatch.setattr(flagcx_mod, "import_module", importer)
    if has_api:
        with pytest.raises(ImportError, match="libcudart"):
            flagcx_mod._import_flagcx_api()
        importer.assert_called_once_with("flagcx.api")
    else:
        assert flagcx_mod._import_flagcx_api() is legacy
        assert "flagcx" not in sys.modules
        importer.assert_called_once_with("plugin.interservice.flagcx_wrapper")
