# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare FlagGems chunk GDN with SGLang's native MUSA path."""

import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA hardware",
)
def test_chunk_gdn_matches_native_with_square_asymmetric_state():
    from sglang_fl.dispatch.backends.flagos.impl.fla import (
        chunk_gated_delta_rule_flagos,
    )
    from sglang_fl.dispatch.fla_patch import get_original

    import sglang_fl

    sglang_fl.load_plugin()
    native = get_original("chunk_gated_delta_rule")
    assert native is not None

    torch.manual_seed(47)
    device, dtype = "musa", torch.bfloat16
    q = torch.randn((1, 64, 1, 64), device=device, dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    g = -torch.rand((1, 64, 1), device=device, dtype=dtype) * 0.1
    beta = torch.rand((1, 64, 1), device=device, dtype=dtype) * 0.5
    # K == V intentionally: a shape check alone cannot detect a transposed state.
    state = torch.arange(64 * 64, device=device, dtype=torch.float32)
    state = state.reshape(1, 1, 64, 64) / (64 * 64)
    state = state + torch.arange(64, device=device)[None, None, :, None] * 0.01
    indices = torch.tensor([0], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 64], device=device, dtype=torch.int32)
    common = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=64**-0.5,
        initial_state_indices=indices, cu_seqlens=cu_seqlens,
        head_first=False, use_qk_l2norm_in_kernel=False,
    )

    native_state, flagos_state = state.clone(), state.clone()
    native_output, _, _ = native(**common, initial_state=native_state)
    flagos_output, _, _ = chunk_gated_delta_rule_flagos(
        **common, initial_state=flagos_state
    )
    torch.musa.synchronize()

    torch.testing.assert_close(flagos_output, native_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(flagos_state, native_state, rtol=3e-2, atol=3e-2)
