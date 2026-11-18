# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA EAGLE runtime patches and the MTP commit-scatter patch (CPU only)."""

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import eagle_runtime as er
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (
    mamba_commit_scatter as mcs,
)


def test_draft_backend_maps_txda_oot_to_triton(monkeypatch) -> None:
    draft_utils = pytest.importorskip("sglang.srt.speculative.draft_utils")
    seen = {}

    def original(self, backend_name, backend_map, error_template, stamps_children=False):
        seen["map"] = backend_map
        return "created"

    monkeypatch.setattr(draft_utils.DraftBackendFactory, "_create_backend", original)
    er._draft_backend()
    fn = draft_utils.DraftBackendFactory._create_backend
    triton_ctor = object()
    assert fn(None, "decode_attention_backend", {"triton": triton_ctor}, "err") == "created"
    assert seen["map"]["txda_oot"] is triton_ctor
    er._draft_backend()  # idempotent
    assert draft_utils.DraftBackendFactory._create_backend is fn


def test_draft_graphs_skipped_on_txda_only(monkeypatch) -> None:
    worker_mod = pytest.importorskip("sglang.srt.speculative.eagle_worker_v2")
    calls = []
    monkeypatch.setattr(worker_mod.EagleDraftWorker, "_capture_cuda_graphs", lambda self: calls.append(self))
    er._draft_graphs()
    fn = worker_mod.EagleDraftWorker._capture_cuda_graphs

    txda = SimpleNamespace(target_worker=SimpleNamespace(device="txda"),
                           cuda_graph_runner="x", cuda_graph_runner_for_draft_extend="y")
    fn(txda)
    assert txda.cuda_graph_runner is None and txda.cuda_graph_runner_for_draft_extend is None
    assert calls == []
    cuda = SimpleNamespace(target_worker=SimpleNamespace(device="cuda"))
    fn(cuda)
    assert calls == [cuda]


def test_dense_conv_windows(monkeypatch) -> None:
    mp = pytest.importorskip("sglang.srt.mem_cache.memory_pool")
    monkeypatch.setattr(mp, "conv_window_dedup_enabled", mp.conv_window_dedup_enabled)
    assert mp.conv_window_dedup_enabled(False, False, 1, False) is True  # upstream: dedup on chains
    er._dense_conv_windows()
    assert mp.conv_window_dedup_enabled(False, False, 1, False) is False


def test_input_logprobs_fast_path_disabled(monkeypatch) -> None:
    lp = pytest.importorskip("sglang.srt.layers.logprob_processor")

    def original(self, *args, **kwargs):
        self.enable_fast_input_logprobs = True

    monkeypatch.setattr(lp.InputLogprobProcessor, "__init__", original)
    er._input_logprobs()
    obj = lp.InputLogprobProcessor.__new__(lp.InputLogprobProcessor)
    lp.InputLogprobProcessor.__init__(obj)
    assert obj.enable_fast_input_logprobs is False


def test_mamba_commit_uses_per_pair_scatter(monkeypatch) -> None:
    m = pytest.importorskip("sglang.kernels.ops.mamba.mamba_state_scatter_triton")
    monkeypatch.setattr(m, "_conv_multi_eligible", lambda pairs: True)
    mcs._apply()
    assert m._conv_multi_eligible([object()]) is False
