"""PPU breakable CUDA graph capture status uses torch, not cuda-python."""

import sys
from types import ModuleType

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.thead.patches import graph_capture_status as gcs


def test_patch_replaces_module_level_query(monkeypatch):
    mod = ModuleType(gcs._MODULE)

    def original(stream):
        raise AssertionError("cuda-python path must not be used on PPU")

    mod._is_stream_capturing = original
    monkeypatch.setitem(sys.modules, gcs._MODULE, mod)

    gcs.patch_graph_capture_status()

    assert mod._is_stream_capturing is gcs._is_stream_capturing


def test_missing_module_is_ignored(monkeypatch):
    monkeypatch.setitem(sys.modules, gcs._MODULE, None)  # import -> ImportError
    gcs.patch_graph_capture_status()


@pytest.mark.gpu
def test_reports_capture_state_on_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA-API device required")
    stream = torch.cuda.Stream()
    assert gcs._is_stream_capturing(stream) is False

    graph = torch.cuda.CUDAGraph()
    x = torch.zeros(4, device="cuda")
    seen = []
    with torch.cuda.graph(graph, stream=stream):
        x.add_(1)
        seen.append(gcs._is_stream_capturing(stream))
    assert seen == [True]
