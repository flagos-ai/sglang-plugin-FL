# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Hygon's public FLA kernel adapters."""

import pytest
import torch

from sglang_fl.dispatch import fla_patch
from sglang_fl.dispatch.backends.vendor.hygon.hcu import hcuBackend
from sglang_fl.dispatch.backends.vendor.hygon.impl import (
    fla_chunk,
    fla_fused_recurrent,
    fla_packed_decode,
)


@pytest.fixture(autouse=True)
def saved_originals(monkeypatch):
    # Isolate the global save-before-patching registry from other vendors.
    originals = {}
    monkeypatch.setattr(fla_patch, "_originals", originals)
    return originals


def test_chunk_preserves_indexed_state_and_public_return(saved_originals):
    indices = torch.tensor([2, 0])
    state = torch.empty(3, 2, 4, 4)
    seq_lengths = torch.tensor([0, 3, 5])
    expected = (object(), None, state)

    def native(**kwargs):
        assert kwargs["initial_state_indices"] is indices
        assert kwargs["initial_state"] is state
        assert kwargs["cu_seqlens"] is seq_lengths
        assert kwargs["scale"] is None
        assert kwargs["use_qk_l2norm_in_kernel"] is True
        return expected

    saved_originals["chunk_gated_delta_rule"] = native
    assert (
        hcuBackend().chunk_gated_delta_rule(
            None,
            None,
            None,
            None,
            None,
            initial_state=state,
            initial_state_indices=indices,
            cu_seqlens=seq_lengths,
            use_qk_l2norm_in_kernel=True,
        )
        is expected
    )


def test_chunk_keeps_public_input_validation(saved_originals):
    def native(**kwargs):
        if kwargs["head_first"]:
            raise DeprecationWarning("head_first is deprecated")

    saved_originals["chunk_gated_delta_rule"] = native
    with pytest.raises(DeprecationWarning, match="head_first"):
        fla_chunk.chunk_gated_delta_rule_hcu(
            None, None, None, None, None, head_first=True
        )


def test_recurrent_current_api_preserves_optional_defaults(saved_originals):
    state, seq_lengths = object(), object()
    expected = (object(), state)

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
        assert beta is None and scale is None
        assert initial_state is state
        assert output_final_state is False
        assert cu_seqlens is seq_lengths
        assert use_qk_l2norm_in_kernel is True
        return expected

    saved_originals["fused_recurrent_gated_delta_rule"] = native
    assert (
        hcuBackend().fused_recurrent_gated_delta_rule(
            None,
            None,
            None,
            None,
            initial_state=state,
            output_final_state=False,
            cu_seqlens=seq_lengths,
            use_qk_l2norm_in_kernel=True,
        )
        is expected
    )


def test_recurrent_legacy_scheduling_is_preserved(saved_originals):
    indices, accepted = object(), object()

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

    saved_originals["fused_recurrent_gated_delta_rule"] = native
    assert fla_fused_recurrent.fused_recurrent_gated_delta_rule_hcu(
        None,
        None,
        None,
        None,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
    ) == (indices, accepted)


@pytest.mark.parametrize("name", ["ssm_state_indices", "num_accepted_tokens"])
def test_recurrent_rejects_unsupported_scheduling_before_launch(saved_originals, name):
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
        pytest.fail("Unsupported scheduling must not launch a state update")

    saved_originals["fused_recurrent_gated_delta_rule"] = native
    with pytest.raises(NotImplementedError, match=name):
        fla_fused_recurrent.fused_recurrent_gated_delta_rule_hcu(
            None, None, None, None, **{name: object()}
        )


def test_packed_decode_preserves_output_and_envelope_strided_state(saved_originals):
    # A page-major state pool need not have a contiguous per-layer slot pitch.
    pool = torch.zeros(3, 2, 2, 4, 8)
    state = pool[:, 0]
    output = torch.empty(2, 1, 2, 4)
    indices = torch.tensor([2, 0])

    def native(**kwargs):
        assert kwargs["initial_state"] is state
        assert kwargs["initial_state"].stride(0) == pool.stride(0)
        assert kwargs["out"] is output
        assert kwargs["ssm_state_indices"] is indices
        assert kwargs["use_qk_l2norm_in_kernel"] is True
        state[indices] = 3
        output.fill_(7)
        return output, state

    saved_originals["fused_recurrent_gated_delta_rule_packed_decode"] = native
    result = fla_packed_decode.fused_recurrent_gated_delta_rule_packed_decode_hcu(
        mixed_qkv=None,
        a=None,
        b=None,
        A_log=None,
        dt_bias=None,
        scale=1.0,
        initial_state=state,
        out=output,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    assert result[0] is output and result[1] is state
    torch.testing.assert_close(output, torch.full_like(output, 7))
    torch.testing.assert_close(pool[2, 0], torch.full_like(pool[2, 0], 3))
    assert torch.count_nonzero(pool[:, 1]) == 0


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: fla_chunk.chunk_gated_delta_rule_hcu(None, None, None, None, None),
        lambda: fla_fused_recurrent.fused_recurrent_gated_delta_rule_hcu(
            None, None, None, None
        ),
        lambda: fla_packed_decode.fused_recurrent_gated_delta_rule_packed_decode_hcu(
            None, None, None, None, None, 1.0, None, None, None
        ),
    ],
)
def test_missing_original_fails_without_importing_patched_public_function(invoke):
    with pytest.raises(RuntimeError, match="initialize the FL plugin"):
        invoke()
