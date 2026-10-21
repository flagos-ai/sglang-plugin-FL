"""CPU contract tests for the optional FlagGems adapters used on empty/CoreX."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl import utils
from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
from sglang_fl.dispatch.backends.flagos.impl import fla, fused_moe
from sglang_fl.dispatch.backends.flagos.register_ops import register_builtins
from sglang_fl.dispatch.registry import OpRegistry


def _reference(q, k, v, g, beta, state, cu, scale, normalize=False):
    """Independent token recurrence in SGLang's [V,K] state layout."""
    output = torch.empty_like(v)
    states_after = []
    qf, kf, vf = (x.reshape(-1, *x.shape[2:]).float() for x in (q, k, v))
    gf = g.reshape(-1, *g.shape[2:]).float()
    bf = beta.reshape(-1, *beta.shape[2:]).float()
    output_flat = output.reshape(-1, *output.shape[2:])
    heads = torch.arange(v.shape[2]) // (v.shape[2] // q.shape[2])
    for seq, (bos, eos) in enumerate(zip(cu[:-1], cu[1:])):
        h = state[seq].float().clone()
        seq_states = []
        for token in range(int(bos), int(eos)):
            qt, kt = qf[token], kf[token]
            if normalize:
                qt = qt / (qt.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                kt = kt / (kt.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            qt, kt = qt[heads], kt[heads]
            h = h * gf[token].exp()[:, None, None]
            delta = vf[token] - torch.einsum("hvk,hk->hv", h, kt)
            weight = bf[token]
            if weight.ndim == 1:
                weight = weight[:, None]
            h += (delta * weight)[:, :, None] * kt[:, None, :]
            output_flat[token] = torch.einsum("hvk,hk->hv", h, qt) * scale
            seq_states.append(h.clone())
        state[seq].copy_(h)
        states_after.append(seq_states)
    return output, state, states_after


def _fake_recurrent(**kwargs):
    """Emulate the two state layouts of FlagGems 5.3.4 on CPU."""
    assert "output_final_state" not in kwargs
    assert kwargs["inplace_final_state"]
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    transpose = kwargs.pop("_state_is_kv", None)
    if transpose is None:
        transpose = q.shape[1] <= 64 or all(
            x.stride(0) == x.stride(1) + x.stride(2) for x in (q, k, v)
        )
    state = kwargs["initial_state"]
    public = state.transpose(-1, -2) if transpose else state
    cu, indices = kwargs["cu_seqlens"], kwargs["ssm_state_indices"]
    selected = []
    for seq in range(len(cu) - 1):
        index = int(indices[seq] if indices.ndim == 1 else indices[seq, 0])
        selected.append(public[max(0, index)].clone())
    selected = torch.stack(selected)
    output, final, _ = _reference(
        q,
        k,
        v,
        kwargs["g"],
        kwargs["beta"],
        selected,
        cu,
        kwargs["scale"],
        kwargs["use_qk_l2norm_in_kernel"],
    )
    for seq in range(len(cu) - 1):
        index = int(indices[seq] if indices.ndim == 1 else indices[seq, 0])
        if index >= 0:
            public[index].copy_(final[seq])
        else:
            output.reshape(-1, *output.shape[2:])[
                int(cu[seq]) : int(cu[seq + 1])
            ].fill_(float("nan"))
    return output, state


class _FakePackedKernel:
    def __init__(self, calls):
        self.calls = calls

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append(kwargs)
            assert kwargs["T"] == 1 and kwargs["INPLACE_FINAL_STATE"]
            assert grid[2] == kwargs["B"] * kwargs["HV"]
            output, _ = _fake_recurrent(
                q=kwargs["q"],
                k=kwargs["k"],
                v=kwargs["v"],
                g=kwargs["g"],
                beta=kwargs["beta"],
                scale=kwargs["scale"],
                initial_state=kwargs["h0"],
                inplace_final_state=True,
                cu_seqlens=kwargs["cu_seqlens"],
                ssm_state_indices=kwargs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=kwargs["USE_QK_L2NORM_IN_KERNEL"],
                _state_is_kv=False,
            )
            valid = (kwargs["ssm_state_indices"] >= 0)[:, None, None, None]
            kwargs["o"].copy_(torch.where(valid, output, kwargs["o"]))

        return launch


def _fake_packed_runtime(calls):
    runtime = SimpleNamespace(
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        cdiv=lambda n, d: (n + d - 1) // d,
    )
    return runtime, _FakePackedKernel(calls)


@pytest.mark.parametrize("length,heads", [(1, 2), (2, 1), (3, 2), (65, 2)])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_recurrent_preserves_public_state_and_optional_final(
    monkeypatch, length, heads, output_final_state
):
    torch.manual_seed(13)
    q, k = (torch.randn(2, length, heads, 4) * 0.1 for _ in range(2))
    v = torch.randn(2, length, heads, 3)
    g = -torch.rand(2, length, heads)
    beta = torch.rand_like(g)
    state = torch.randn(2, heads, 3, 4)
    original = state.clone()
    monkeypatch.setattr(fla, "_recurrent_kernel", lambda: _fake_recurrent)
    actual, final = fla.fused_recurrent_gated_delta_rule_flagos(
        q,
        k,
        v,
        g,
        beta,
        0.5,
        state,
        output_final_state,
    )
    expected, expected_final, _ = _reference(
        q,
        k,
        v,
        g,
        beta,
        original.clone(),
        torch.tensor([0, length, 2 * length]),
        0.5,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, original)
    if output_final_state:
        torch.testing.assert_close(final, expected_final)
    else:
        assert final is None


def test_recurrent_variable_length_and_zero_initial_state(monkeypatch):
    q, k = (torch.rand(1, 5, 2, 4) * 0.1 for _ in range(2))
    v = torch.rand(1, 5, 4, 3)
    g = -torch.ones(1, 5, 4)
    beta = torch.rand_like(g)
    cu = torch.tensor([0, 2, 5])
    monkeypatch.setattr(fla, "_recurrent_kernel", lambda: _fake_recurrent)
    actual, final = fla.fused_recurrent_gated_delta_rule_flagos(
        q, k, v, g, beta, 0.5, cu_seqlens=cu, output_final_state=True
    )
    expected, expected_final, _ = _reference(
        q, k, v, g, beta, torch.zeros(2, 4, 3, 4), cu, 0.5
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(final, expected_final)


def _mock_chunk_pipeline(monkeypatch):
    captured = {}

    def cumsum(**kwargs):
        assert kwargs["chunk_size"] == 64
        captured.update(kwargs)
        return kwargs["g"], object()

    def recompute(**kwargs):
        return kwargs["k"], kwargs["v"]

    def recurrent(**kwargs):
        assert kwargs["chunk_size"] == 64
        assert kwargs["output_final_state"]
        captured["state"] = kwargs["initial_state"]
        return torch.empty(1, 1, 1, 4, 3), kwargs["u"], captured["final"]

    def output(**kwargs):
        return captured["output"]

    monkeypatch.setattr(
        fla, "_chunk_kernels", lambda: (cumsum, recompute, recurrent, output)
    )
    return captured


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_transposes_states_scatter_and_h_contract(monkeypatch, head_first):
    q, k = (torch.randn(1, 3, 1, 4) for _ in range(2))
    v = torch.randn(1, 3, 1, 3)
    g = -torch.rand(1, 3, 1)
    beta = torch.rand_like(g)
    state = torch.arange(48).reshape(4, 1, 3, 4).float()
    original = state.clone()
    indices = torch.tensor([2, -1, 0])
    captured = _mock_chunk_pipeline(monkeypatch)
    captured["final"] = torch.arange(36).reshape(3, 1, 4, 3).float()
    captured["output"] = v
    args = (q, k, v, g, beta)
    if head_first:
        args = tuple(x.transpose(1, 2) for x in args)
    actual, final, h = fla.chunk_gated_delta_rule_flagos(
        *args,
        0.5,
        state,
        indices,
        torch.tensor([0, 1, 2, 3]),
        head_first,
        True,
    )
    expected_selected = original[indices.clamp_min(0)].transpose(-1, -2).contiguous()
    expected_selected[1].zero_()
    torch.testing.assert_close(captured["state"], expected_selected)
    torch.testing.assert_close(captured["k"], fla._l2norm(k))
    torch.testing.assert_close(state[2], captured["final"][0].transpose(-1, -2))
    torch.testing.assert_close(state[0], captured["final"][2].transpose(-1, -2))
    torch.testing.assert_close(state[1], original[1])
    torch.testing.assert_close(state[3], original[3])
    assert final is None and h.shape == (1, 1, 1, 3, 4)
    torch.testing.assert_close(actual, v.transpose(1, 2) if head_first else v)


@pytest.mark.parametrize("batch", [1, 3, 65])
def test_packed_decode_independent_rows_fp32_gates_and_inplace_cache(
    monkeypatch, batch
):
    torch.manual_seed(123)
    heads, value_heads, key_dim, value_dim = 1, 2, 4, 3
    mixed = torch.randn(
        batch, 2 * heads * key_dim + value_heads * value_dim, dtype=torch.float16
    )
    a, b = (torch.randn(batch, value_heads, dtype=torch.float16) for _ in range(2))
    A_log, dt_bias = torch.rand(value_heads), torch.rand(value_heads)
    state = torch.randn(batch + 1, value_heads, value_dim, key_dim)
    original = state.clone()
    indices = torch.arange(batch - 1, -1, -1)
    if batch > 1:
        indices[-1] = -1
    out = torch.full((batch, 1, value_heads, value_dim), 7.0, dtype=mixed.dtype)
    calls = []

    monkeypatch.setattr(
        fla, "_packed_decode_kernel", lambda: _fake_packed_runtime(calls)
    )
    actual_out, actual_state = (
        fla.fused_recurrent_gated_delta_rule_packed_decode_flagos(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            0.5,
            state,
            out,
            indices,
            True,
        )
    )
    assert actual_out is out and actual_state is state
    assert calls[0]["q"].shape == (batch, 1, heads, key_dim)
    assert calls[0]["g"].dtype == calls[0]["beta"].dtype == torch.float32
    q, k, v = mixed.split(
        (heads * key_dim, heads * key_dim, value_heads * value_dim), -1
    )
    q, k = (x.reshape(batch, 1, heads, key_dim) for x in (q, k))
    v = v.reshape(batch, 1, value_heads, value_dim)
    g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    selected = original[indices.clamp_min(0)].clone()
    expected, final, _ = _reference(
        q,
        k,
        v,
        g[:, None],
        b.float().sigmoid()[:, None],
        selected,
        torch.arange(batch + 1),
        0.5,
        True,
    )
    for row, index in enumerate(indices):
        if index >= 0:
            torch.testing.assert_close(out[row], expected[row])
            torch.testing.assert_close(state[index], final[row])
        else:
            torch.testing.assert_close(out[row], torch.zeros_like(out[row]))
    torch.testing.assert_close(state[-1], original[-1])


def _moe_inputs():
    config = SimpleNamespace(
        activation="silu",
        is_gated=True,
        no_combine=False,
        num_experts=2,
        num_local_experts=2,
        apply_router_weight_on_input=True,
        routed_scaling_factor=2.0,
    )
    obj = SimpleNamespace(moe_runner_config=config)
    layer = SimpleNamespace(
        w13_weight=torch.rand(2, 8, 3),
        w2_weight=torch.rand(2, 3, 4),
        w13_weight_bias=torch.rand(2, 8),
        w2_weight_bias=torch.rand(2, 3),
    )
    topk = SimpleNamespace(
        topk_weights=torch.rand(5, 2), topk_ids=torch.ones(5, 2, dtype=torch.int64)
    )
    dispatch = SimpleNamespace(hidden_states=torch.rand(5, 3), topk_output=topk)
    return obj, layer, dispatch


def test_moe_forwards_bias_router_weight_scaling_and_combine_contract(monkeypatch):
    obj, layer, dispatch = _moe_inputs()
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(kwargs["hidden_states"])

    fused = ModuleType("flag_gems.fused")
    fused.fused_experts_impl = kernel
    combine = ModuleType("sglang.srt.layers.moe.token_dispatcher.standard")
    combine.StandardCombineInput = SimpleNamespace
    monkeypatch.setitem(sys.modules, "flag_gems.fused", fused)
    monkeypatch.setitem(sys.modules, combine.__name__, combine)
    result = fused_moe.fused_moe_flagos(obj, layer, dispatch)
    assert calls[0]["topk_ids"].dtype == torch.int32
    assert calls[0]["inplace"] is False
    assert calls[0]["apply_router_weight_on_input"] is True
    assert calls[0]["w1_bias"] is layer.w13_weight_bias
    assert calls[0]["w2_bias"] is layer.w2_weight_bias
    torch.testing.assert_close(
        result.hidden_states, torch.full_like(dispatch.hidden_states, 2.0)
    )


@pytest.mark.parametrize(
    "target,name,value",
    [
        ("obj", "use_triton_kernels", True),
        ("obj", "w13_swiglu_interleaved", True),
        ("obj", "use_fp8_w8a8", True),
        ("obj", "quant_config", object()),
        ("config", "activation", "gelu"),
        ("config", "no_combine", True),
        ("config", "num_local_experts", 1),
        ("config", "swiglu_limit", 7.0),
        ("config", "gemm1_beta", 1.0),
        ("dispatch", "hidden_states_pre_quant", (1, 2)),
    ],
)
def test_moe_rejects_unsupported_semantics_before_kernel(target, name, value):
    obj, layer, dispatch = _moe_inputs()
    setattr(
        {"obj": obj, "config": obj.moe_runner_config, "dispatch": dispatch}[target],
        name,
        value,
    )
    with pytest.raises(NotImplementedError):
        fused_moe.fused_moe_flagos(obj, layer, dispatch)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_moe_rejects_quantized_weights_before_importing_kernel(dtype):
    obj, layer, dispatch = _moe_inputs()
    layer.w13_weight = layer.w13_weight.to(dtype)
    layer.w2_weight = layer.w2_weight.to(dtype)
    with pytest.raises(NotImplementedError, match="unquantized"):
        fused_moe.fused_moe_flagos(obj, layer, dispatch)


def test_optional_kernel_capabilities_are_probed_per_operator(monkeypatch):
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name="iluvatar")
    )
    monkeypatch.setattr(FlagOSBackend, "_available", True)
    monkeypatch.setattr(torch, "corex", object(), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        fla, "_chunk_kernels", lambda: (lambda: None, None, lambda: None, lambda: None)
    )
    monkeypatch.setattr(
        fla, "_recurrent_kernel", lambda: lambda output_final_state=True: None
    )
    monkeypatch.setattr(fla, "_packed_decode_kernel", lambda: (None, None))
    monkeypatch.setattr(fused_moe, "is_fused_moe_available", lambda: False)
    registry = OpRegistry()
    register_builtins(registry)
    impls = registry.snapshot().impls_by_op
    assert impls["silu_and_mul"][0].is_available()
    for name in (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "fused_moe",
    ):
        assert not impls[name][0].is_available()
    monkeypatch.setattr(fla, "_chunk_kernels", lambda: (lambda: None,) * 4)
    monkeypatch.setattr(
        fla, "_recurrent_kernel", lambda: lambda inplace_final_state=True: None
    )
    monkeypatch.setattr(fla, "_packed_decode_kernel", lambda: _fake_packed_runtime([]))
    assert impls["chunk_gated_delta_rule"][0].is_available()
    assert impls["fused_recurrent_gated_delta_rule"][0].is_available()
    assert impls["fused_recurrent_gated_delta_rule_packed_decode"][0].is_available()


@pytest.mark.parametrize(
    "vendor_name,recurrent_available",
    [("nvidia", True), ("mthreads", True), ("ascend", False)],
)
def test_new_flagos_defaults_do_not_replace_other_vendors(
    monkeypatch, vendor_name, recurrent_available
):
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name=vendor_name)
    )
    monkeypatch.setattr(FlagOSBackend, "_available", True)
    monkeypatch.delattr(torch, "corex", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fla, "is_fla_op_available", lambda name: True)
    monkeypatch.setattr(fused_moe, "is_fused_moe_available", lambda: True)
    backend = FlagOSBackend()
    registry = OpRegistry()
    register_builtins(registry)
    impls = registry.snapshot().impls_by_op
    for name in (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "fused_moe",
    ):
        assert not backend.is_op_available(name)
        assert not impls[name][0].is_available()
    assert (
        backend.is_op_available("fused_recurrent_gated_delta_rule")
        is recurrent_available
    )
    assert (
        impls["fused_recurrent_gated_delta_rule"][0].is_available()
        is recurrent_available
    )


def test_recurrent_default_omits_final_state(monkeypatch):
    q, k = (torch.ones(1, 1, 1, 4) * 0.1 for _ in range(2))
    v = torch.ones(1, 1, 1, 3)
    g, beta = -torch.ones(1, 1, 1), torch.full((1, 1, 1), 0.5)
    monkeypatch.setattr(fla, "_recurrent_kernel", lambda: _fake_recurrent)
    output, final_state = fla.fused_recurrent_gated_delta_rule_flagos(
        q, k, v, g, beta, 0.5
    )
    assert output.shape == v.shape
    assert final_state is None


def test_backend_recurrent_forwards_false_default(monkeypatch):
    calls = []
    sentinel = object()

    def implementation(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(fla, "fused_recurrent_gated_delta_rule_flagos", implementation)
    result = FlagOSBackend().fused_recurrent_gated_delta_rule(
        object(), object(), object(), object(), object(), 0.5
    )
    assert result is sentinel
    assert calls[0][7] is False
