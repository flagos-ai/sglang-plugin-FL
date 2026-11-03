import os

import torch

from sglang_fl.dispatch.manager import OpManager
from sglang_fl.dispatch.policy import SelectionPolicy, policy_context
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _initialized_manager(registry: OpRegistry) -> OpManager:
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    return manager


def test_mhc_builtins_register_flagos_and_reference_implementations():
    from sglang_fl.dispatch.backends.flagos.register_ops import (
        register_builtins as register_flagos,
    )
    from sglang_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )

    registry = OpRegistry()
    register_flagos(registry)
    register_reference(registry)

    for op_name in ("mhc_pre", "mhc_post"):
        assert {impl.impl_id for impl in registry.get_implementations(op_name)} == {
            "default.flagos",
            "reference.torch",
        }


def test_mhc_reference_pre_and_post_preserve_contract():
    from sglang_fl.dispatch.backends.reference.impl.mhc import (
        mhc_post_torch,
        mhc_pre_torch,
    )

    torch.manual_seed(7)
    tokens, streams, hidden = 3, 4, 8
    residual = torch.randn(tokens, streams, hidden, dtype=torch.bfloat16)
    fn = torch.randn(2 * streams + streams * streams, streams * hidden)
    scale = torch.tensor([0.01, 0.01, 0.01])
    base = torch.randn(2 * streams + streams * streams)

    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    assert post_mix.shape == (tokens, streams, 1)
    assert comb_mix.shape == (tokens, streams, streams)
    assert layer_input.shape == (tokens, hidden)
    projection = residual.float().flatten(-2, -1)
    inv_rms = (projection.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    mixes = (projection @ fn.float().T) * inv_rms
    expanded_scale = torch.cat(
        (
            scale[0].expand(streams),
            scale[1].expand(streams),
            scale[2].expand(streams * streams),
        )
    )
    mixes = mixes * expanded_scale + base
    expected_pre = mixes[:, :streams].sigmoid().unsqueeze(-1) + 1e-6
    expected_post = mixes[:, streams : 2 * streams].sigmoid().unsqueeze(-1) * 2.0
    expected_comb = (mixes[:, 2 * streams :].reshape(-1, streams, streams)).softmax(-1) + 1e-6
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        expected_comb = expected_comb / (
            expected_comb.sum(-1, keepdim=True) + 1e-6
        )
        expected_comb = expected_comb / (
            expected_comb.sum(-2, keepdim=True) + 1e-6
        )
    expected_layer = (residual.float() * expected_pre).sum(-2).to(residual.dtype)
    torch.testing.assert_close(post_mix, expected_post)
    torch.testing.assert_close(comb_mix, expected_comb)
    torch.testing.assert_close(layer_input, expected_layer)
    torch.testing.assert_close(
        comb_mix.sum(dim=-2), torch.ones(tokens, streams), atol=2e-4, rtol=2e-4
    )

    output = mhc_post_torch(layer_input, residual, post_mix, comb_mix)
    expected = layer_input.float().unsqueeze(-2) * post_mix
    expected += torch.matmul(comb_mix.transpose(-1, -2), residual.float())
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)


def test_mhc_fake_outputs_match_graph_contract():
    from sglang_fl.dispatch.bridge.mhc import mhc_pre_fake

    residual = torch.empty(5, 4, 8, dtype=torch.bfloat16, device="meta")
    outputs = mhc_pre_fake(
        residual,
        torch.empty(24, 32, device="meta"),
        torch.empty(3, device="meta"),
        torch.empty(24, device="meta"),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        8,
    )
    assert [tensor.shape for tensor in outputs] == [(5, 4, 1), (5, 4, 4), (5, 8)]
    assert [tensor.dtype for tensor in outputs] == [
        torch.float32,
        torch.float32,
        torch.bfloat16,
    ]
    assert all(tensor.device.type == "meta" for tensor in outputs)


def test_selected_mhc_failure_is_not_retried_and_is_counted():
    registry = OpRegistry()
    calls = {"primary": 0, "reference": 0}
    def primary(value):
        calls["primary"] += 1
        raise RuntimeError(f"kernel failed for {value}")

    def reference(value):
        calls["reference"] += 1
        return f"reference:{value}"

    registry.register_many(
        [
            OpImpl(
                op_name="mhc_pre",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=primary,
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                op_name="mhc_pre",
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=reference,
                priority=BackendPriority.REFERENCE,
            ),
        ]
    )
    manager = _initialized_manager(registry)

    with policy_context(SelectionPolicy.from_dict(strict=False)):
        try:
            manager.call_selected("mhc_pre", "shape")
        except RuntimeError as exc:
            assert str(exc) == "kernel failed for shape"
        else:
            raise AssertionError("runtime failure must propagate")

    assert calls == {"primary": 1, "reference": 0}
    stats = manager.get_selected_call_stats("mhc_pre")["mhc_pre"]
    assert stats == {
        "calls": 1,
        "successes": 0,
        "failures": 1,
        "implementations": {
            "default.flagos": {"attempts": 1, "successes": 0, "failures": 1},
        },
    }
