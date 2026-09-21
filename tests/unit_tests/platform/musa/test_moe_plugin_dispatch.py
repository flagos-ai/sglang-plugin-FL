# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.impl import fused_moe as entry
from sglang_fl.dispatch.backends.vendor.mthreads.moe import dispatch
from sglang_fl.dispatch.backends.vendor.mthreads.patches import moe_workspace


def _tensor(shape, dtype=torch.bfloat16, device="musa", contiguous=True):
    return NS(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=NS(type=device, index=0),
        is_contiguous=lambda: contiguous,
    )


def _contract():
    config = NS(
        num_experts=256,
        num_local_experts=256,
        num_fused_shared_experts=0,
        hidden_size=2048,
        intermediate_size_per_partition=256,
        top_k=8,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        no_combine=False,
    )
    layer = NS(
        moe_runner_config=config,
        w13_weight=_tensor((256, 512, 2048)),
        w2_weight=_tensor((256, 2048, 256)),
        moe_tp_size=2,
    )
    method = NS(
        _aiter_runner=None,
        with_bias=False,
        moe_runner_config=config,
        runner=NS(runner_backend=NS(is_triton=lambda: True)),
    )
    args = NS(disable_piecewise_cuda_graph=True)
    a2a = NS(is_none=lambda: True)
    return method, layer, args, a2a


def test_layer_check_does_not_mutate_or_duplicate_weights():
    method, layer, args, a2a = _contract()
    weights = (layer.w13_weight, layer.w2_weight)
    before = dict(vars(layer))
    assert dispatch._matches_layer(method, layer, args, a2a)
    assert dispatch._matches_layer(method, layer, args, a2a)
    assert vars(layer) == before
    assert weights == (layer.w13_weight, layer.w2_weight)
    assert not hasattr(method, "fuse_swiglu_epilogue")


@pytest.mark.parametrize(
    "scope,field,value",
    [
        ("method", "_aiter_runner", object()),
        ("method", "with_bias", True),
        ("layer", "moe_tp_size", 1),
        ("layer", "w13_weight_bias", object()),
        ("config", "num_local_experts", 128),
        ("config", "top_k", 4),
        ("config", "num_fused_shared_experts", 1),
        ("config", "activation", "gelu"),
        ("config", "is_gated", False),
        ("config", "no_combine", True),
        ("config", "apply_router_weight_on_input", True),
        ("config", "gemm1_alpha", 1.7),
        ("config", "gemm1_clamp_limit", 7.0),
        ("args", "enable_lora", True),
        ("args", "lora_paths", ["adapter"]),
        ("args", "enable_eplb", True),
        ("args", "enable_fused_moe_sum_all_reduce", True),
        ("args", "disable_piecewise_cuda_graph", False),
        ("runner", "lora_enabled", True),
        ("runner", "down_gemm_overlap_args", object()),
        ("runner", "meta_overlap_args", {}),
    ],
)
def test_layer_guard_misses(scope, field, value):
    method, layer, args, a2a = _contract()
    targets = dict(
        method=method,
        layer=layer,
        args=args,
        config=layer.moe_runner_config,
        runner=method.runner,
    )
    setattr(targets[scope], field, value)
    assert not dispatch._matches_layer(method, layer, args, a2a)


@pytest.mark.parametrize(
    "which", ["w13", "w2", "dtype", "layout", "a2a", "runner", "config"]
)
def test_noncanonical_layer_retains_fallback(which):
    method, layer, args, a2a = _contract()
    if which == "w13":
        layer.w13_weight.shape = (256, 1024, 2048)
    elif which == "w2":
        layer.w2_weight.shape = (128, 2048, 256)
    elif which == "dtype":
        layer.w13_weight.dtype = torch.float16
    elif which == "layout":
        layer.w2_weight.is_contiguous = lambda: False
    elif which == "a2a":
        a2a.is_none = lambda: False
    elif which == "runner":
        method.runner.runner_backend.is_triton = lambda: False
    else:
        method.moe_runner_config = NS(**vars(layer.moe_runner_config))
    assert not dispatch._matches_layer(method, layer, args, a2a)


@pytest.mark.parametrize(
    "miss",
    [
        None,
        "empty",
        "cuda",
        "shape",
        "ids_dtype",
        "weights_dtype",
        "device",
        "layout",
        "grad",
    ],
)
def test_runtime_tensor_guards(miss):
    _, layer, _, _ = _contract()
    hidden, ids, weights = (
        _tensor((64, 2048)),
        _tensor((64, 8), torch.int32),
        _tensor((64, 8), torch.float32),
    )
    if miss == "empty":
        hidden.shape, ids.shape, weights.shape = (0, 2048), (0, 8), (0, 8)
    elif miss == "cuda":
        hidden.device.type = "cuda"
    elif miss == "shape":
        hidden.shape = (64, 4096)
    elif miss == "ids_dtype":
        ids.dtype = torch.int64
    elif miss == "weights_dtype":
        weights.dtype = torch.bfloat16
    elif miss == "device":
        weights.device.index = 1
    elif miss == "layout":
        ids.is_contiguous = lambda: False
    with torch.set_grad_enabled(miss == "grad"):
        assert dispatch._matches_inputs(layer, hidden, weights, ids) == (miss is None)


def test_dispatch_falls_back_exactly_once_and_preserves_return(monkeypatch):
    monkeypatch.delenv("SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE", raising=False)
    monkeypatch.delenv("SGLANG_MUSA_M4_W13_BN64", raising=False)
    calls, sentinel = [], object()
    obj = NS(forward_musa=lambda *a: calls.append(a) or sentinel)
    layer, output = object(), object()
    assert entry.fused_moe_musa(obj, layer, output) is sentinel
    assert calls == [(layer, output)]


def test_cpu_falls_back_even_when_requested(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE", "1")
    assert (
        dispatch.maybe_forward(
            object(), object(), NS(hidden_states=torch.zeros(4, 2048))
        )
        is None
    )


def test_dispatch_hit_and_kernel_error_are_not_retried(monkeypatch):
    def forbidden(*args):
        pytest.fail("native fallback must not run after a candidate hit")

    obj = NS(forward_musa=forbidden)
    sentinel = object()
    monkeypatch.setattr(dispatch, "maybe_forward", lambda *a: sentinel)
    assert entry.fused_moe_musa(obj, object(), object()) is sentinel
    error = RuntimeError("launch failed")

    def fail(*args):
        raise error

    monkeypatch.setattr(dispatch, "maybe_forward", fail)
    with pytest.raises(RuntimeError) as exc:
        entry.fused_moe_musa(obj, object(), object())
    assert exc.value is error


def test_workspace_precedes_memory_measurement_and_preserves_parent(monkeypatch):
    events = []

    class Parent:
        def init_memory_pool(self, pre_model_load_memory):
            events.append(("pool", pre_model_load_memory))
            return 123

    class Runner(Parent):
        pass

    original = Parent.init_memory_pool
    monkeypatch.setenv("SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE", "1")
    monkeypatch.setattr(
        moe_workspace, "_reserve", lambda r: events.append(("reserve", r))
    )
    assert moe_workspace._patch_runner(Runner)
    first = Runner.init_memory_pool
    assert moe_workspace._patch_runner(Runner)
    assert Runner.init_memory_pool is first
    assert Parent.init_memory_pool is original
    runner = Runner()
    assert runner.init_memory_pool(pre_model_load_memory=99) == 123
    assert events == [("reserve", runner), ("pool", 99)]
    events.clear()
    monkeypatch.setenv("SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE", "0")
    runner.init_memory_pool(77)
    assert events == [("pool", 77)]


def test_workspace_failure_stops_before_kv_allocation(monkeypatch):
    class Runner:
        def init_memory_pool(self, pre_model_load_memory):
            pytest.fail("KV allocation must not proceed after workspace OOM")

    def oom(runner):
        raise RuntimeError("out of memory")

    monkeypatch.setenv("SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE", "1")
    monkeypatch.setattr(moe_workspace, "_reserve", oom)
    assert moe_workspace._patch_runner(Runner)
    with pytest.raises(RuntimeError, match="out of memory"):
        Runner().init_memory_pool(99)


def test_workspace_unknown_abi_does_not_patch():
    class Runner:
        def init_memory_pool(self, memory, new_required_parameter):
            pass

    original = Runner.init_memory_pool
    assert not moe_workspace._patch_runner(Runner)
    assert Runner.init_memory_pool is original
