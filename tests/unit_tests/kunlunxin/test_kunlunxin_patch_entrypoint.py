# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Contract test for ``vendor/kunlunxin/patch.py``.

``patch.py`` applies every Kunlunxin monkey-patch at import time, so this test
imports it against stubbed ``patches.*`` modules: that keeps the production
ordering guarantee (entrypoint applied on import) while never mutating another
vendor's sglang in the same pytest process.
"""

import importlib
import sys
import types

import pytest

PKG = "sglang_fl.dispatch.backends.vendor.kunlunxin"
PATCH_MODULE = f"{PKG}.patch"

# Order matters: PDL-like helpers must precede consumers (see patch.py).
EXPECTED_CALLS = (
    ("clamp_position", "patch_clamp_position"),
    ("causal_conv1d", "patch_causal_conv1d"),
    ("suppress_pynccl", "patch_suppress_pynccl"),
    ("pp_send_first", "patch_pp_send_recv_and_preprocess_output_tensors"),
    ("attention_backend_choice", "patch_attention_backend_choice"),
    ("qwen35_attn_native", "patch_qwen35_attn_native"),
    ("gdn_fused_qkv_split", "patch_gdn_fused_qkv_split"),
    ("vocab_embed_native", "patch_vocab_embed_native"),
)


@pytest.fixture
def entrypoint(monkeypatch):
    calls = []

    for module_name, fn_name in EXPECTED_CALLS:
        module = types.ModuleType(f"{PKG}.patches.{module_name}")

        def _record(name=fn_name):
            calls.append(name)

        setattr(module, fn_name, _record)
        monkeypatch.setitem(sys.modules, f"{PKG}.patches.{module_name}", module)

    # Import the entrypoint into a private module object: it is executed once
    # per process in production, and this test must not leave a stubbed copy
    # cached in sys.modules for the rest of the session.
    monkeypatch.delitem(sys.modules, PATCH_MODULE, raising=False)
    module = importlib.import_module(PATCH_MODULE)
    try:
        yield module, calls
    finally:
        sys.modules.pop(PATCH_MODULE, None)


def test_entrypoint_applies_every_patch_on_import(entrypoint):
    module, calls = entrypoint
    assert calls == [fn for _, fn in EXPECTED_CALLS]
    assert module._patches_applied is True


def test_entrypoint_is_idempotent(entrypoint):
    module, calls = entrypoint
    applied = list(calls)
    module.apply_kunlunxin_patches()
    assert calls == applied


def test_patch_functions_exist_in_the_real_modules():
    """Guard against a rename in ``patches/`` that patch.py does not follow."""
    for module_name, fn_name in EXPECTED_CALLS:
        real = importlib.import_module(f"{PKG}.patches.{module_name}")
        assert callable(getattr(real, fn_name)), (module_name, fn_name)


def test_every_patch_module_is_covered_by_this_test_matrix():
    """The vendor dir is the coverage contract: each patch file is listed here."""
    from pathlib import Path

    patches_dir = (
        Path(importlib.import_module(PKG).__file__).resolve().parent / "patches"
    )
    on_disk = {
        path.stem
        for path in patches_dir.glob("*.py")
        if path.stem != "__init__"
    }
    assert on_disk == {module for module, _ in EXPECTED_CALLS}
