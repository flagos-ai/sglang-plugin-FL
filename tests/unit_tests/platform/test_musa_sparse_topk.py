# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor/mthreads/patches/sparse_topk.py"
)
spec = importlib.util.spec_from_file_location("sparse_topk_patch_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prunes_invalid_tiles_without_mutating_original_tuner():
    configs = [NS(kwargs={"BLOCK_SIZE_K": k}) for k in (64, 128, 256)]
    tuner = NS(early_config_prune=None, configs=configs)
    original = NS(fn=tuner)
    upstream = NS(_topk_index_kernel=original)
    assert module.install_sparse_topk_patch(upstream)
    prune = upstream._topk_index_kernel.fn.early_config_prune
    args = {"s_ptr": NS(device=NS(type="musa")), "topk": 64}
    assert [x.kwargs["BLOCK_SIZE_K"] for x in prune(configs, args)] == [128, 256]
    assert tuner.early_config_prune is None
    args["s_ptr"].device.type = "cuda"
    assert prune(configs, args) is configs
    assert not module.install_sparse_topk_patch(upstream)


def test_no_legal_tile_fails_before_launch():
    upstream = NS(_topk_index_kernel=NS(fn=NS(early_config_prune=None, configs=[])))
    module.install_sparse_topk_patch(upstream)
    with pytest.raises(NotImplementedError, match="No legal"):
        upstream._topk_index_kernel.fn.early_config_prune(
            [NS(kwargs={"BLOCK_SIZE_K": 64})],
            {"s_ptr": NS(device=NS(type="musa")), "topk": 64},
        )


def test_missing_prune_api_preserves_original():
    original = object()
    upstream = NS(_topk_index_kernel=original)
    with pytest.raises(RuntimeError, match="pruning API"):
        module.install_sparse_topk_patch(upstream)
    assert upstream._topk_index_kernel is original
