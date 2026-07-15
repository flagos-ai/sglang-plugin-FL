# Copyright (c) 2026 BAAI. All rights reserved.

"""Functional smoke tests for SGLang-FL graph capture integration.

SGLang-FL does not provide its own graph wrapper. It declares graph capability
through ``PlatformFL`` and lets SGLang choose the native GraphRunner. These tests
therefore cover platform graph capability, runner selection, basic CUDA graph
capture/replay, and a small ``call_op`` capture smoke path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.functional]


def _platform_for(device_type: str):
    from sglang_fl.platform import PlatformFL

    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = device_type
    return platform


def test_platform_graph_capability_flags() -> None:
    """Check PlatformFL graph capability declarations by device type."""
    expected = {
        "cuda": (True, True),
        "npu": (True, False),
        "musa": (True, False),
        "cpu": (False, False),
    }

    for device_type, (cuda_graph, piecewise_graph) in expected.items():
        platform = _platform_for(device_type)
        assert platform.support_cuda_graph() is cuda_graph
        assert platform.support_piecewise_cuda_graph() is piecewise_graph


def test_cuda_like_platform_uses_sglang_cuda_graph_runner() -> None:
    """CUDA and MUSA reuse SGLang's native CudaGraphRunner."""
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    assert _platform_for("cuda").get_graph_runner_cls() is CudaGraphRunner
    assert _platform_for("musa").get_graph_runner_cls() is CudaGraphRunner


def test_npu_platform_uses_sglang_npu_graph_runner() -> None:
    """NPU selects SGLang's native NPUGraphRunner when the NPU package exists."""
    try:
        from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
            NPUGraphRunner,
        )
    except Exception as exc:  # pragma: no cover - depends on NPU dependencies
        pytest.skip(f"SGLang NPUGraphRunner unavailable in this environment: {exc}")

    assert _platform_for("npu").get_graph_runner_cls() is NPUGraphRunner


@pytest.mark.gpu
def test_basic_cuda_graph_capture_and_replay(device) -> None:
    """Check basic CUDA graph capture/replay works in the functional environment."""
    if device.type != "cuda":
        pytest.skip("CUDA graph capture smoke test requires CUDA")

    static_input = torch.randn(4, 8, device=device)
    static_output = torch.empty_like(static_input)

    def computation(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x * 2 + 1)

    computation(static_input, static_output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        computation(static_input, static_output)

    new_input = torch.ones_like(static_input)
    static_input.copy_(new_input)
    graph.replay()

    assert torch.allclose(static_output, new_input * 2 + 1)


@pytest.mark.gpu
def test_call_op_silu_and_mul_cuda_graph_capture(device) -> None:
    """Check FL dispatch/call_op does not break a small CUDA graph capture."""
    if device.type != "cuda":
        pytest.skip("call_op graph capture smoke test requires CUDA")

    from sglang_fl.dispatch import SelectionPolicy, call_op, policy_context

    static_input = torch.randn(4, 16, device=device)
    static_output = torch.empty(4, 8, device=device)
    policy = SelectionPolicy.from_dict(prefer="reference", strict=False)

    with policy_context(policy):
        static_output.copy_(call_op("silu_and_mul", None, static_input))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with policy_context(policy):
        with torch.cuda.graph(graph):
            static_output.copy_(call_op("silu_and_mul", None, static_input))

    new_input = torch.randn_like(static_input)
    static_input.copy_(new_input)
    graph.replay()

    half = new_input.shape[-1] // 2
    expected = F.silu(new_input[..., :half]) * new_input[..., half:]
    assert torch.allclose(static_output, expected, rtol=1e-3, atol=1e-3)
