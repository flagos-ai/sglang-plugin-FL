import torch
import pytest

from sglang_fl.dispatch.backends.flagos.impl import mhc as flagos_mhc
from sglang_fl.dispatch.backends.reference.impl.mhc import (
    mhc_post_torch,
    mhc_pre_torch,
)
from sglang_fl.dispatch.backends.reference.register_ops import register_builtins
from sglang_fl.dispatch.registry import OpRegistry


def _inputs(dtype=torch.float32):
    torch.manual_seed(7)
    tokens, hc_mult, hidden = 3, 2, 4
    residual = torch.randn(tokens, hc_mult, hidden, dtype=dtype)
    mix_width = (2 + hc_mult) * hc_mult
    fn = torch.randn(mix_width, hc_mult * hidden, dtype=torch.float32)
    scale = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32)
    base = torch.randn(mix_width, dtype=torch.float32)
    return residual, fn, scale, base


def test_mhc_pre_matches_contract():
    residual, fn, scale, base = _inputs()
    pre_eps = 0.125
    sinkhorn_eps = 1e-4
    post_mult = 1.5
    repeat = 3

    post, comb, layer_input = mhc_pre_torch(
        residual,
        fn,
        scale,
        base,
        1e-6,
        pre_eps,
        sinkhorn_eps,
        post_mult,
        repeat,
    )

    flat = residual.flatten(1)
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    mixes = torch.nn.functional.linear(normalized, fn)
    hc_mult = residual.shape[1]
    expected_pre = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult])
    expected_pre = expected_pre + pre_eps
    expected_post = torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * scale[1]
        + base[hc_mult : 2 * hc_mult]
    ) * post_mult
    expected_comb = (
        mixes[:, 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
    ).reshape(-1, hc_mult, hc_mult)
    expected_comb = torch.exp(
        expected_comb - expected_comb.amax(dim=-1, keepdim=True)
    )
    expected_comb = (
        expected_comb / expected_comb.sum(dim=-1, keepdim=True) + sinkhorn_eps
    )
    expected_comb = expected_comb / (
        expected_comb.sum(dim=-2, keepdim=True) + sinkhorn_eps
    )
    for _ in range(repeat - 1):
        expected_comb = expected_comb / (
            expected_comb.sum(dim=-1, keepdim=True) + sinkhorn_eps
        )
        expected_comb = expected_comb / (
            expected_comb.sum(dim=-2, keepdim=True) + sinkhorn_eps
        )

    expected_layer_input = (residual * expected_pre.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(post.squeeze(-1), expected_post)
    torch.testing.assert_close(comb, expected_comb)
    torch.testing.assert_close(layer_input, expected_layer_input)
    assert post.dtype == torch.float32
    assert comb.dtype == torch.float32
    assert layer_input.dtype == residual.dtype


def test_mhc_post_matches_contract():
    residual, _, _, _ = _inputs()
    x = torch.randn(residual.shape[0], residual.shape[2])
    post = torch.randn(residual.shape[0], residual.shape[1], 1)
    comb = torch.randn(residual.shape[0], residual.shape[1], residual.shape[1])

    actual = mhc_post_torch(x, residual, post, comb)
    expected = x.unsqueeze(1) * post + torch.bmm(comb.mT, residual)

    torch.testing.assert_close(actual, expected)


def test_mhc_pre_keeps_reference_math_in_float32_for_bfloat16_input():
    residual, fn, scale, base = _inputs(dtype=torch.bfloat16)

    post, comb, layer_input = mhc_pre_torch(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 4
    )

    flat = residual.float().flatten(1)
    reciprocal_rms = torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-6
    )
    mixes = torch.nn.functional.linear(flat, fn) * reciprocal_rms
    hc_mult = residual.shape[1]
    expected_pre = torch.sigmoid(
        mixes[:, :hc_mult] * scale[0] + base[:hc_mult]
    ) + 1e-6
    expected_post = 2.0 * torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * scale[1]
        + base[hc_mult : 2 * hc_mult]
    )
    expected_comb = (
        mixes[:, 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
    ).reshape(-1, hc_mult, hc_mult)
    expected_comb = expected_comb.softmax(-1) + 1e-6
    expected_comb = expected_comb / (
        expected_comb.sum(-2, keepdim=True) + 1e-6
    )
    for _ in range(3):
        expected_comb = expected_comb / (
            expected_comb.sum(-1, keepdim=True) + 1e-6
        )
        expected_comb = expected_comb / (
            expected_comb.sum(-2, keepdim=True) + 1e-6
        )
    expected_input = (
        residual.float() * expected_pre.unsqueeze(-1)
    ).sum(dim=1).to(torch.bfloat16)

    torch.testing.assert_close(post.squeeze(-1), expected_post)
    torch.testing.assert_close(comb, expected_comb)
    torch.testing.assert_close(layer_input, expected_input)
    assert post.dtype == torch.float32
    assert comb.dtype == torch.float32
    assert layer_input.dtype == torch.bfloat16


def test_mhc_reference_handles_empty_batch():
    residual, fn, scale, base = _inputs()
    outputs = mhc_pre_torch(
        residual[:0], fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    assert [tuple(output.shape) for output in outputs] == [
        (0, 2, 1),
        (0, 2, 2),
        (0, 4),
    ]


@pytest.mark.parametrize(
    "pre_eps,sinkhorn_eps,post_mult,error",
    [
        (1e-5, 1e-6, 2.0, "hc_pre_eps == hc_sinkhorn_eps"),
        (1e-6, 1e-6, 1.5, "hc_post_mult_value == 2.0"),
    ],
)
def test_flagos_mhc_rejects_unsupported_parameters(
    pre_eps, sinkhorn_eps, post_mult, error
):
    with pytest.raises(NotImplementedError, match=error):
        flagos_mhc._validate_mhc_pre_contract(
            pre_eps, sinkhorn_eps, post_mult
        )


def test_flagos_mhc_requires_accelerator_operands():
    with pytest.raises(RuntimeError, match="accelerator operands"):
        flagos_mhc._require_accelerator(torch.empty(1))


def test_flagos_mhc_pre_matches_reference_with_flaggems_torch_path(monkeypatch):
    residual, fn, scale, base = _inputs(dtype=torch.float32)
    monkeypatch.setattr(flagos_mhc, "_require_accelerator", lambda tensor: None)

    actual = flagos_mhc.mhc_pre_flagos(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 4
    )
    expected = mhc_pre_torch(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 4
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_reference_backend_registers_mhc_ops():
    registry = OpRegistry()
    register_builtins(registry)

    assert registry.get_implementation("mhc_pre", "reference.torch") is not None
    assert registry.get_implementation("mhc_post", "reference.torch") is not None
