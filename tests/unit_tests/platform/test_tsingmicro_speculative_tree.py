# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA EAGLE/MTP tree patch: SGLang 0.5.18 contract and rebinding (no device work)."""

import pytest

eagle_utils = pytest.importorskip("sglang.srt.speculative.eagle_utils")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    speculative_tree as st,
)


def test_upstream_triton_fallbacks_match_expected_contract() -> None:
    assert st._signature_is(eagle_utils.sgl_build_tree_kernel_triton, st._TREE_PARAMETERS)
    assert st._signature_is(eagle_utils.verify_tree_greedy_triton, st._VERIFY_PARAMETERS)
    assert st._signature_is(
        eagle_utils.verify_tree_greedy_func, (*st._VERIFY_PARAMETERS, "topk")
    )


def test_apply_rebinds_build_and_verify(monkeypatch) -> None:
    calls = []

    def fake_verify_triton(**kwargs):
        calls.append(sorted(kwargs))

    monkeypatch.setattr(
        eagle_utils, "sgl_build_tree_kernel_efficient", object(), raising=False
    )
    monkeypatch.setattr(
        eagle_utils, "verify_tree_greedy_func", eagle_utils.verify_tree_greedy_func
    )
    st._apply()

    assert eagle_utils.sgl_build_tree_kernel_efficient is eagle_utils.sgl_build_tree_kernel_triton
    original = eagle_utils.verify_tree_greedy_func.__wrapped__
    assert original.__module__ == eagle_utils.__name__  # wrapper around the upstream func

    monkeypatch.setattr(st, "_signature_is", lambda *a, **k: True)
    monkeypatch.setattr(eagle_utils, "verify_tree_greedy_triton", fake_verify_triton)
    st._apply()
    tensors = [object() for _ in range(8)]
    out = eagle_utils.verify_tree_greedy_func(*tensors, topk=1)
    assert out == (tensors[0], tensors[1], tensors[2])
    assert calls == [sorted(st._VERIFY_PARAMETERS)]


def test_signature_mismatch_leaves_module_untouched(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(eagle_utils, "sgl_build_tree_kernel_efficient", sentinel, raising=False)
    monkeypatch.setattr(eagle_utils, "sgl_build_tree_kernel_triton", lambda x: x)
    st._apply()
    assert eagle_utils.sgl_build_tree_kernel_efficient is sentinel
