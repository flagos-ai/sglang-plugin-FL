# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the FlagCX device communicator wrapper."""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


class _FakeFlagCXLibrary:
    def __init__(self, library_path: str) -> None:
        self.library_path = library_path


class _FakeFlagCXId:
    def __init__(self) -> None:
        self.internal = [0] * 128


def _write_wrapper(path: Path, marker: str, *, prelude: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        prelude
        + f"""
MARKER = {marker!r}
FLAGCXLibrary = object()
buffer_type = object()
flagcxStream_t = object()
flagcxDataTypeEnum = object()
flagcxRedOpTypeEnum = object()
flagcxUniqueId = object()
"""
    )


def _fake_wrapper_module():
    return SimpleNamespace(
        FLAGCXLibrary=_FakeFlagCXLibrary,
        buffer_type=lambda ptr=None: ptr,
        flagcxComm_t=object,
        flagcxStream_t=object,
        flagcxDataTypeEnum=object,
        flagcxRedOpTypeEnum=object,
        flagcxUniqueId=_FakeFlagCXId,
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


def test_import_flagcx_wrapper_loads_new_layout_without_package_init(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    package_dir = tmp_path / "src" / "flagcx"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "raise RuntimeError('flagcx package initializer must not run')\n"
    )
    api_path = package_dir / "api.py"
    _write_wrapper(
        api_path,
        "new",
        prelude=(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Function:\n"
            "    name: str\n"
        ),
    )
    _write_wrapper(
        tmp_path / "plugin" / "interservice" / "flagcx_wrapper.py", "legacy"
    )
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    module = flagcx_mod._import_flagcx_wrapper()

    assert module.MARKER == "new"
    assert Path(module.__file__) == api_path


def test_import_flagcx_wrapper_falls_back_to_legacy_layout(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    wrapper_path = (
        tmp_path / "plugin" / "interservice" / "flagcx_wrapper.py"
    )
    _write_wrapper(wrapper_path, "legacy")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    module = flagcx_mod._import_flagcx_wrapper()

    assert module.MARKER == "legacy"
    assert Path(module.__file__) == wrapper_path


def test_import_flagcx_wrapper_ignores_cached_foreign_package(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    api_path = tmp_path / "src" / "flagcx" / "api.py"
    _write_wrapper(api_path, "configured")
    foreign_package = ModuleType("flagcx")
    foreign_package.__path__ = ["/foreign/flagcx"]
    foreign_api = ModuleType("flagcx.api")
    foreign_api.MARKER = "foreign"
    monkeypatch.setitem(flagcx_mod.sys.modules, "flagcx", foreign_package)
    monkeypatch.setitem(flagcx_mod.sys.modules, "flagcx.api", foreign_api)
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    module = flagcx_mod._import_flagcx_wrapper()

    assert module.MARKER == "configured"
    assert Path(module.__file__) == api_path


def test_import_flagcx_wrapper_uses_installed_package(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    installed_api = _fake_wrapper_module()
    import_module = Mock(return_value=installed_api)
    monkeypatch.setattr(flagcx_mod.importlib, "import_module", import_module)

    assert flagcx_mod._import_flagcx_wrapper() is installed_api
    import_module.assert_called_once_with("flagcx.api")


def test_import_flagcx_wrapper_does_not_hide_dependency_error(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    dependency_missing = ModuleNotFoundError("No module named 'torch_npu'")
    dependency_missing.name = "torch_npu"
    import_module = Mock(side_effect=dependency_missing)
    monkeypatch.setattr(flagcx_mod.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="torch_npu"):
        flagcx_mod._import_flagcx_wrapper()

    import_module.assert_called_once_with("flagcx.api")


def test_import_flagcx_wrapper_cleans_up_failed_source_module(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    api_path = tmp_path / "src" / "flagcx" / "api.py"
    api_path.parent.mkdir(parents=True)
    api_path.write_text("raise RuntimeError('wrapper failed')\n")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    modules_before = {
        name
        for name in flagcx_mod.sys.modules
        if name.startswith(flagcx_mod._SOURCE_MODULE_PREFIX)
    }

    with pytest.raises(RuntimeError, match="wrapper failed"):
        flagcx_mod._import_flagcx_wrapper()

    modules_after = {
        name
        for name in flagcx_mod.sys.modules
        if name.startswith(flagcx_mod._SOURCE_MODULE_PREFIX)
    }
    assert modules_after == modules_before


def test_import_flagcx_wrapper_validates_required_symbols(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    api_path = tmp_path / "src" / "flagcx" / "api.py"
    api_path.parent.mkdir(parents=True)
    api_path.write_text("FLAGCXLibrary = object()\n")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    modules_before = {
        name
        for name in flagcx_mod.sys.modules
        if name.startswith(flagcx_mod._SOURCE_MODULE_PREFIX)
    }

    with pytest.raises(ImportError, match="buffer_type"):
        flagcx_mod._import_flagcx_wrapper()

    modules_after = {
        name
        for name in flagcx_mod.sys.modules
        if name.startswith(flagcx_mod._SOURCE_MODULE_PREFIX)
    }
    assert modules_after == modules_before


def test_import_flagcx_wrapper_reuses_source_module(
    monkeypatch, tmp_path: Path
) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    _write_wrapper(tmp_path / "src" / "flagcx" / "api.py", "new")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))

    first = flagcx_mod._import_flagcx_wrapper()
    second = flagcx_mod._import_flagcx_wrapper()

    assert second is first


def test_world_size_one_disables_communicator(monkeypatch, tmp_path: Path) -> None:
    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    monkeypatch.setattr(flagcx_mod, "_import_flagcx_wrapper", _fake_wrapper_module)
    monkeypatch.setattr(flagcx_mod.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(flagcx_mod.dist, "get_world_size", Mock(return_value=1))

    comm = flagcx_mod.FlagCXCommunicator(group="group", device=torch.device("cpu"))

    assert comm.disabled is True
    assert comm.available is False
    assert comm.group == "group"


def test_get_stream_uses_imported_stream_type() -> None:
    from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

    stream_type = Mock(return_value=ctypes.c_void_p())
    comm = FlagCXCommunicator.__new__(FlagCXCommunicator)
    comm._stream_cache = {}
    comm._stream_type = stream_type
    comm.flagcx = Mock()
    comm.flagcx.devHandle.contents.streamCopy.return_value = 0
    comm._get_raw_stream_handle = Mock(return_value=123)

    stream = comm._get_stream()

    assert stream is comm._stream_cache[123]
    stream_type.assert_called_once_with()
    comm.flagcx.devHandle.contents.streamCopy.assert_called_once()


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
