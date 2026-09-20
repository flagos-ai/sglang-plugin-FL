# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned-image TopK/combine hit and input-refresh checks; no model weights."""

import inspect

import pytest
import torch

from tests.functional_tests.compilation.test_graph_capture import _graph_api

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.fixture
def musa_device(device):
    if (
        device.type != "musa"
        or "S5000" not in torch.musa.get_device_name(device).upper()
    ):
        pytest.skip("requires the pinned MUSA S5000 runtime")
    return device


@pytest.mark.parametrize("tokens", [4, 64, 2048])
def test_real_topk_guard_launch_and_refreshed_replay(musa_device, monkeypatch, tokens):
    import triton
    from sglang.srt.hardware_backend.musa.kernels import topk as musa_topk

    from sglang_fl.dispatch.backends.vendor.mthreads.patches import topk_schedule

    graph_cls, capture, synchronize = _graph_api(musa_device)
    kernel = musa_topk.topk_softmax_triton_kernel
    ctx = topk_schedule._verify_pinned_startup(
        kernel, triton.Config({}, num_warps=1, num_stages=1)
    )
    assert ctx is not None, "real runtime objects did not pass the pinned TopK guard"
    original = inspect.unwrap(musa_topk.topk_softmax)
    calls = []
    launch = topk_schedule._pinned_launch_closed

    def observed_launch(*args):
        calls.append(True)
        return launch(*args)

    monkeypatch.setattr(topk_schedule, "_pinned_launch_closed", observed_launch)
    wrapped = topk_schedule._make_topk_wrapper(original, kernel, ctx)
    gating = torch.empty((tokens, 256), dtype=torch.float32, device=musa_device)
    weights = torch.empty((tokens, 8), dtype=torch.float32, device=musa_device)
    ids = torch.empty((tokens, 8), dtype=torch.int32, device=musa_device)
    generator = torch.Generator().manual_seed(20260920)
    phases = [
        torch.randn(tokens, 256, generator=generator).to(musa_device) for _ in range(2)
    ]
    gating.copy_(phases[0])
    wrapped(weights, ids, gating, True)
    assert calls, "TopK silently used the original path"
    synchronize()
    graph = graph_cls()
    with capture(graph):
        wrapped(weights, ids, gating, True)
    pointers = gating.data_ptr(), weights.data_ptr(), ids.data_ptr()
    for phase in (0, 1, 0):
        gating.copy_(phases[phase])
        expected_weights, expected_ids = (
            torch.empty_like(weights),
            torch.empty_like(ids),
        )
        original(expected_weights, expected_ids, gating, True)
        wrapped(weights, ids, gating, True)
        eager_weights, eager_ids = weights.clone(), ids.clone()
        before_replay = len(calls)
        for _ in range(3):
            graph.replay()
            synchronize()
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(ids, eager_ids, rtol=0, atol=0)
            torch.testing.assert_close(weights, eager_weights, rtol=0, atol=0)
        assert len(calls) == before_replay, "replay unexpectedly reran Python dispatch"
        assert pointers == (gating.data_ptr(), weights.data_ptr(), ids.data_ptr())


@pytest.mark.parametrize("tokens", [40, 64, 2048])
def test_real_combine_hit_fallback_and_refreshed_replay(
    musa_device, monkeypatch, tokens
):
    from sglang_fl.dispatch.backends.vendor.mthreads.patches import moe_combine

    monkeypatch.setenv(moe_combine._ENV_NAME, "auto")
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_DECODE_GRAPH_CANDIDATE_DISABLED", False)
    graph_cls, capture, synchronize = _graph_api(musa_device)
    routed = torch.empty((tokens, 8, 2048), dtype=torch.bfloat16, device=musa_device)
    output = torch.empty((tokens, 2048), dtype=torch.bfloat16, device=musa_device)
    shared = torch.empty_like(output)
    gate = torch.empty((tokens, 1), dtype=torch.bfloat16, device=musa_device)
    dual_stream = tokens in (40, 64)
    primary = torch.musa.current_stream()
    alternate = torch.musa.Stream() if dual_stream else primary
    context = moe_combine.MoeCombineContext(
        shared,
        gate,
        decode_graph_dual_stream=dual_stream,
        shared_stream=primary,
    )
    fallback_calls = []

    def original(*args):
        fallback_calls.append(True)
        output.zero_()

    reduction = moe_combine._wrap_moe_sum_reduce(original)

    def execute():
        context.used = False
        token = moe_combine._ACTIVE_CONTEXT.set(context)
        try:
            if dual_stream:
                alternate.wait_stream(primary)
                with torch.musa.stream(alternate):
                    reduction(routed, output, 1.0)
                primary.wait_stream(alternate)
            else:
                reduction(routed, output, 1.0)
        finally:
            moe_combine._ACTIVE_CONTEXT.reset(token)

    # Dyadic values plus exact sigmoid endpoints isolate indexing, reduction,
    # stream ownership and refresh from approximation differences in exp().
    pattern = (torch.arange(2048, device=musa_device) % 17).to(torch.bfloat16)

    def refresh(phase):
        routed.copy_(pattern.view(1, 1, -1).expand_as(routed))
        routed.mul_(phase + 1)
        shared.fill_(2 + phase)
        gate.fill_(0 if phase == 0 else float("inf"))
        return (
            (pattern.float() * (8 * (phase + 1)) + (1 if phase == 0 else 3))
            .to(torch.bfloat16)
            .expand_as(output)
        )

    refresh(0)
    execute()
    assert context.used and not fallback_calls, "combine silently used fallback"
    synchronize()
    graph = graph_cls()
    with capture(graph):
        execute()
    assert context.used and not fallback_calls
    pointers = tuple(t.data_ptr() for t in (routed, output, shared, gate))
    for phase in (0, 1, 0):
        expected = refresh(phase)
        execute()
        assert context.used and not fallback_calls
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        eager = output.clone()
        for _ in range(3):
            graph.replay()
            synchronize()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
        assert pointers == tuple(t.data_ptr() for t in (routed, output, shared, gate))
    monkeypatch.setenv(moe_combine._ENV_NAME, "off")
    execute()
    synchronize()
    assert not context.used and fallback_calls == [True]
    assert torch.count_nonzero(output).item() == 0
    assert moe_combine._ACTIVE_CONTEXT.get() is None
