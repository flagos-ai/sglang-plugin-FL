# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA fallback dispatch and recurrent FLA API compatibility."""

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla


def _invoke(**kwargs):
    return fla.fused_recurrent_gated_delta_rule_musa(
        q=None, k=None, v=None, g=None, beta=None, scale=1.0, **kwargs
    )


def test_recurrent_current_api(monkeypatch):
    def native(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    ):
        return initial_state, output_final_state

    monkeypatch.setattr(fla, "_original", lambda name: native)
    state = object()
    assert _invoke(initial_state=state) == (state, True)


def test_recurrent_legacy_state_indices_are_preserved(monkeypatch):
    def native(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        ssm_state_indices,
        num_accepted_tokens,
    ):
        return ssm_state_indices, num_accepted_tokens

    monkeypatch.setattr(fla, "_original", lambda name: native)
    indices, accepted = object(), object()
    assert _invoke(ssm_state_indices=indices, num_accepted_tokens=accepted) == (
        indices,
        accepted,
    )


@pytest.mark.parametrize("argument", ["ssm_state_indices", "num_accepted_tokens"])
def test_recurrent_rejects_unsupported_state_indexing(monkeypatch, argument):
    def native(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    ):
        pytest.fail("Unsupported state indexing must fail before invoking the kernel")

    monkeypatch.setattr(fla, "_original", lambda name: native)
    with pytest.raises(NotImplementedError, match=argument):
        _invoke(**{argument: object()})


def test_unbridged_fused_op_uses_musa_forward(monkeypatch):
    pytest.importorskip("sglang.kernels.fused_op")
    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.srt.platforms import current_platform
    from sglang_fl.platform import PlatformFL

    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = "mthreads"
    platform._device_type = "musa"
    monkeypatch.setattr(
        current_platform, "get_dispatch_key_name", platform.get_dispatch_key_name
    )
    monkeypatch.setattr(current_platform, "is_out_of_tree", lambda: True)

    class MusaOnlyOp(BaseFusedOp):
        def forward_native(self, x):
            pytest.fail("An unbridged MUSA op must retain its vendor implementation")

        def forward_musa(self, x):
            return x + 1

    torch.testing.assert_close(MusaOnlyOp()(torch.tensor(2)), torch.tensor(3))
