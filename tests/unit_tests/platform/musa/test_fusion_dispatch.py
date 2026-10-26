# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
import torch

from sglang_fl.dispatch import SelectionPolicy, policy_context
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    custom_allreduce_rmsnorm as norm_patch,
)
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    moe_combine,
)
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    shared_expert_gate_tail as shared_patch,
)


@pytest.mark.parametrize(
    "op", ["shared_expert_gate_tail", "moe_sum_reduce", "allreduce_rms_norm"]
)
@pytest.mark.parametrize(
    "policy",
    [
        SelectionPolicy(prefer="reference"),
        SelectionPolicy(deny_vendors=frozenset({"mthreads"})),
        SelectionPolicy(allow_vendors=frozenset({"ascend"})),
    ],
)
def test_fusions_obey_common_selection_policy(musa_dispatch, op, policy):
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert musa_dispatch.resolve(op)._is_available()
    with policy_context(policy):
        fn = musa_dispatch.resolve(op)
        assert fn.__module__.endswith("reference.impl.fusions")


def test_shared_bridge_reference_policy_keeps_materialization(
    musa_dispatch, monkeypatch
):
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import shared_expert_gate_tail

    monkeypatch.setattr(
        shared_expert_gate_tail,
        "shared_expert_gate_tail_musa",
        lambda *a: pytest.fail("vendor denied"),
    )
    hidden = torch.zeros((4, 2048), dtype=torch.bfloat16)
    calls = []
    block = NS(
        shared_expert=lambda x: calls.append(x) or torch.full_like(x, 2),
        shared_expert_gate=lambda x: torch.zeros((4, 1), dtype=x.dtype),
    )
    with policy_context(SelectionPolicy(prefer="reference")):
        assert torch.equal(
            shared_patch._shared_forward(block, hidden), torch.ones_like(hidden)
        )
    assert len(calls) == 1


def test_combine_bridge_reference_does_not_consume_shared_context(
    musa_dispatch, monkeypatch
):
    monkeypatch.setattr(
        moe_combine, "_launch_candidate", lambda *a: pytest.fail("vendor denied")
    )
    context = NS(used=False)
    calls = []
    original = lambda *a, **kw: calls.append((a, kw)) or "native"
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        with policy_context(SelectionPolicy(deny_vendors=frozenset({"mthreads"}))):
            assert (
                moe_combine._wrap_moe_sum_reduce(original)(
                    "routed", "out", 1.0, extra=True
                )
                == "native"
            )
        assert calls == [(("routed", "out", 1.0), {"extra": True})]
        assert not context.used
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)


def test_norm_bridge_reference_keeps_collective_and_residual_order(
    musa_dispatch, monkeypatch
):
    events = []
    group = NS(
        world_size=2,
        fused_allreduce_rmsnorm=lambda *a: pytest.fail("vendor denied"),
        all_reduce=lambda x: events.append(("reduce", x)) or "reduced",
    )
    norm = NS(forward=lambda *a: events.append(("norm", *a)) or "done")
    monkeypatch.setattr(norm_patch, "_select_group", lambda _: group)
    with policy_context(SelectionPolicy(prefer="reference")):
        assert (
            norm_patch._forward_with_musa_allreduce_fusion(
                norm, "x", "residual", "post", "weight"
            )
            == "done"
        )
    assert events == [("reduce", "x"), ("norm", "reduced", "residualpost", None)]


def test_failed_fused_collective_is_not_retried(musa_dispatch, monkeypatch):
    calls = []

    def fail(*args):
        calls.append("fused")
        raise RuntimeError("collective failed")

    group = NS(
        world_size=2,
        fused_allreduce_rmsnorm=fail,
        all_reduce=lambda x: pytest.fail("must not repeat collective"),
    )
    norm = NS(variance_epsilon=1e-6, forward=lambda *a: pytest.fail("must propagate"))
    monkeypatch.setattr(norm_patch, "_select_group", lambda _: group)
    with pytest.raises(RuntimeError, match="collective failed"):
        norm_patch._forward_with_musa_allreduce_fusion(
            norm, "x", "residual", None, "weight"
        )
    assert calls == ["fused"]


def test_per_op_order_overrides_vendor_preference(musa_dispatch):
    policy = SelectionPolicy.from_dict(
        prefer="vendor", per_op_order={"allreduce_rms_norm": ["reference", "vendor"]}
    )
    with policy_context(policy):
        assert musa_dispatch.resolve("allreduce_rms_norm").__module__.endswith(
            "reference.impl.fusions"
        )
        assert musa_dispatch.resolve("moe_sum_reduce")._is_available()
