"""Exercise vendor orchestration on CPU with only native kernel calls replaced."""

import ast
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("no_combine", [False, True])
def test_vendor_moe_preserves_shared_expert_input(no_combine):
    path = ROOT / "sglang_fl/dispatch/backends/vendor/enflame/impl/fused_moe.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_experts_impl_gcu"
    )
    torch.manual_seed(42)
    hidden = torch.randn(3, 4)
    original = hidden.clone()
    w1, w2 = torch.randn(2, 6, 4), torch.randn(2, 4, 3)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])

    def kernel(
        a,
        b,
        c,
        _as,
        _bs,
        _bz,
        tw,
        ti,
        _sorted,
        _experts,
        _padded,
        weighted,
        top_k,
        config,
        **kwargs,
    ):
        result = c.view(3, 2, b.shape[1])
        for i in range(3):
            for j in range(2):
                x = a[i] if b is w1 else a.view(3, 2, -1)[i, j]
                result[i, j].copy_(b[int(ti[i, j])] @ x * (tw[i, j] if weighted else 1))

    namespace = dict(
        torch=torch,
        F=F,
        functools=functools,
        Optional=Optional,
        List=List,
        Dict=Dict,
        Any=Any,
        _use_aiter=False,
        _is_cuda=True,
        _is_hip=False,
        _down_moe_use_tma=lambda: False,
        get_default_config=lambda *a, **k: {},
        moe_align_block_size_gcu=lambda *a: (None, None, None),
        invoke_fused_moe_kernel_gcu=kernel,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    actual = namespace[fn.name](hidden, w1, w2, weights, ids, no_combine=no_combine)
    contributions = []
    for i in range(3):
        row = []
        for j in range(2):
            gate, up = (w1[int(ids[i, j])] @ original[i]).chunk(2)
            row.append((w2[int(ids[i, j])] @ (F.silu(gate) * up)) * weights[i, j])
        contributions.append(torch.stack(row))
    expected = torch.stack(contributions)
    torch.testing.assert_close(actual, expected if no_combine else expected.sum(1))
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    assert actual.data_ptr() != hidden.data_ptr()
    # The shared expert must still see the original hidden states.
    torch.testing.assert_close(F.silu(hidden), F.silu(original), rtol=0, atol=0)


def _moe_function(name, **globals_):
    path = ROOT / "sglang_fl/dispatch/backends/vendor/enflame/impl/fused_moe.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_)
    return globals_[name]


def _runner_config(**overrides):
    from types import SimpleNamespace

    values = dict(num_experts=8, num_local_experts=4, no_combine=False,
                  inplace=False, activation="silu", is_gated=True,
                  apply_router_weight_on_input=False, routed_scaling_factor=1.7,
                  gemm1_alpha=2.3, gemm1_clamp_limit=3.1)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("inplace", [False, True])
def test_experts_wrapper_preserves_argument_slots_and_returns_output(inplace):
    import inspect

    signature = inspect.signature(_moe_function("fused_experts_impl_gcu"))
    hidden, w1, w2, weights, ids, b1, b2, output = (object() for _ in range(8))
    # Distinct sentinels catch shifts after the newly inserted no_combine slot.
    optional = {name: object() for name in (
        "w1_scale", "w2_scale", "w1_zp", "w2_zp", "a1_scale", "a2_scale", "block_shape"
    )}
    captured = []

    def implementation(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.append(bound.arguments)
        return output

    wrapper = _moe_function("fused_experts_gcu", fused_experts_impl_gcu=implementation)
    config = _runner_config(inplace=inplace)
    assert wrapper(hidden, w1, w2, (weights, ids, object()), config,
                   b1=b1, b2=b2, **optional) is output
    assert len(captured) == 1
    args = captured[0]
    for name, value in dict(hidden_states=hidden, w1=w1, w2=w2,
                            topk_weights=weights, topk_ids=ids, b1=b1, b2=b2,
                            **optional).items():
        assert args[name] is value
    assert args["inplace"] is inplace
    assert args["no_combine"] is False
    assert args["routed_scaling_factor"] == 1.7
    assert args["gemm1_alpha"] == 2.3
    assert args["gemm1_limit"] == 3.1
    assert args["filter_expert"] is True


def test_experts_wrapper_rejects_unsupported_no_combine_before_kernel():
    from unittest.mock import Mock

    implementation = Mock()
    wrapper = _moe_function("fused_experts_gcu", fused_experts_impl_gcu=implementation)
    with pytest.raises(AssertionError, match="no combine"):
        wrapper(object(), object(), object(), (object(), object(), object()),
                _runner_config(no_combine=True))
    implementation.assert_not_called()


def test_moe_dispatch_packages_the_experts_output():
    from types import SimpleNamespace

    hidden, weights, ids, output, w1, w2 = (object() for _ in range(6))
    config = _runner_config()
    calls = []

    def quant_info(**kwargs):
        assert kwargs["w13_weight"] is w1 and kwargs["w2_weight"] is w2
        return SimpleNamespace(**kwargs, use_fp8_w8a8=False, use_int8_w8a8=False,
                               use_int8_w8a16=False, use_int4_w4a16=False,
                               per_channel_quant=False, w13_scale=None, w2_scale=None,
                               w13_zp=None, w2_zp=None, a13_scale=None, a2_scale=None,
                               block_shape=None)

    def experts(**kwargs):
        calls.append(kwargs)
        return output

    wrapper = _moe_function("fused_moe_gcu", TritonMoeQuantInfo=quant_info,
                            fused_experts_gcu=experts, StandardCombineInput=SimpleNamespace)
    dispatch = SimpleNamespace(hidden_states=hidden, topk_output=(weights, ids, None))
    result = wrapper(SimpleNamespace(runner=SimpleNamespace(config=config)),
                     SimpleNamespace(w13_weight=w1, w2_weight=w2), dispatch)
    assert result.hidden_states is output
    assert len(calls) == 1
    assert calls[0]["hidden_states"] is hidden
    assert calls[0]["topk_output"] is dispatch.topk_output
    assert calls[0]["moe_runner_config"] is config
