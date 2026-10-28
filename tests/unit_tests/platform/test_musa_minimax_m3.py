# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "mthreads"
    / "patches"
    / "minimax_m3.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_musa_minimax_m3_patch", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PATCH_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PATCH_MODULE)
install_minimax_m3_patches = _PATCH_MODULE.install_minimax_m3_patches


def _fake_module():
    class KVCacheConfigurator:
        def _build_oot_mha_kv_pool(self, *, max_total_num_tokens: int):
            return ("oot", max_total_num_tokens)

        def _build_minimax_sparse_kv_pool(self, *, max_total_num_tokens: int):
            return ("minimax", max_total_num_tokens)

    return SimpleNamespace(
        KVCacheConfigurator=KVCacheConfigurator,
        is_minimax_sparse=lambda config: getattr(config, "sparse", False),
    )


def test_minimax_kv_pool_patch_is_scoped_and_idempotent():
    kv_module = _fake_module()
    assert install_minimax_m3_patches(kv_module)
    assert not install_minimax_m3_patches(kv_module)

    obj = kv_module.KVCacheConfigurator()
    obj.device = "musa:0"
    obj.model_config = SimpleNamespace(hf_config=SimpleNamespace(sparse=True))
    assert obj._build_oot_mha_kv_pool(max_total_num_tokens=128) == (
        "minimax",
        128,
    )

    obj.model_config.hf_config.sparse = False
    assert obj._build_oot_mha_kv_pool(max_total_num_tokens=128) == ("oot", 128)

    obj.device = "cuda:0"
    obj.model_config.hf_config.sparse = True
    assert obj._build_oot_mha_kv_pool(max_total_num_tokens=128) == ("oot", 128)


def test_minimax_patch_requires_upstream_capabilities():
    kv_module = SimpleNamespace(KVCacheConfigurator=object)
    assert not install_minimax_m3_patches(kv_module)


def test_minimax_patch_rejects_signature_drift():
    class KVCacheConfigurator:
        def _build_oot_mha_kv_pool(self, token_count):
            return token_count

        def _build_minimax_sparse_kv_pool(self, *, max_total_num_tokens: int):
            return max_total_num_tokens

    kv_module = SimpleNamespace(
        KVCacheConfigurator=KVCacheConfigurator,
        is_minimax_sparse=lambda config: True,
    )

    with pytest.raises(RuntimeError, match="max_total_num_tokens"):
        install_minimax_m3_patches(kv_module)
