# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for Ascend ViT position parity and replay ownership."""

import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


_PATCH_FILE = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor/ascend/patches/vision_graph.py"
)


@pytest.fixture
def patch_module():
    # Import the implementation directly: do not initialize the platform or
    # package's other hardware dispatch paths in these CPU-only tests.
    spec = importlib.util.spec_from_file_location(
        "ascend_vision_graph_contract", _PATCH_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NpuCpuDevice(str):
    """Route through the NPU branch while executing actual CPU arithmetic."""

    type = "npu"


def _install_fake_upstream(monkeypatch, *, bad_prepare=False, bad_replay=False):
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.models",
        "sglang.srt.models.qwen3_vl",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.graph_runner",
        "sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner",
    )
    modules = {name: ModuleType(name) for name in names}

    class Vision:
        if bad_prepare:

            def _prepare_graph_inputs(self, renamed):
                return renamed

        else:

            def _prepare_graph_inputs(self, x, grid_thw):
                self.delegated.append((x, grid_thw))
                return "original-device-path"

        def forward(self, x, grid_thw):
            return "original-eager"

        def patch_embed(self, value):
            self.calls.append("patch_embed")
            return value * 2

        def _use_vectorized_pos_embed(self, count):
            self.calls.append(("vectorization_decision", count))
            return self.vectorized

        def _positions(self, grid, bias):
            tokens = sum(t * h * w for t, h, w in grid)
            values = torch.arange(tokens * 3, dtype=torch.float32).reshape(tokens, 3)
            return values.to(self.dtype) / 32 + bias

        def fast_pos_embed_interpolate_from_list(self, grid):
            self.calls.append("from_list")
            return self._positions(grid, 1)

        def fast_pos_embed_interpolate_vectorized(self, grid):
            self.calls.append("vectorized")
            return self._positions(grid, 2)

        def fast_pos_embed_interpolate(self, grid):
            raise AssertionError("NPU parity must not use the legacy interpolation")

        def rot_pos_emb(self, grid):
            self.calls.append(("rotary", grid))
            return self.cos, self.sin

    class Runner:
        if bad_replay:

            def replay(self, renamed):
                return renamed

        else:

            def replay(
                self,
                graph_key,
                x_3d,
                rotary_pos_emb_cos=None,
                rotary_pos_emb_sin=None,
                output_indices=None,
            ):
                self.calls.append(
                    (
                        graph_key,
                        x_3d,
                        rotary_pos_emb_cos,
                        rotary_pos_emb_sin,
                        output_indices,
                    )
                )
                # Model the real runner's persistent output, overwritten on
                # every same-shape replay, rather than returning new tensors.
                self.static.copy_(x_3d.reshape_as(self.static))
                if output_indices is not None:
                    return self.static.index_select(0, output_indices)
                return self.static

    def cumulative_lengths(grid):
        assert isinstance(grid, torch.Tensor)
        per_frame = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).to(
            torch.int32
        )
        return torch.cat(
            (torch.zeros(1, dtype=torch.int32), per_frame.cumsum(0, dtype=torch.int32))
        )

    modules["sglang.srt.models.qwen3_vl"].Qwen3VLMoeVisionModel = Vision
    modules[
        "sglang.srt.models.qwen3_vl"
    ].compute_cu_seqlens_from_grid_numpy = cumulative_lengths
    modules[
        "sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner"
    ].ViTNpuGraphRunner = Runner
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)
    return Vision, Runner


def _vision(cls, *, dtype=torch.float32, align=False, vectorized=False):
    instance = cls()
    instance.dtype = dtype
    instance.device = _NpuCpuDevice("cpu")
    instance.align_corners = align
    instance.vectorized = vectorized
    instance.calls, instance.delegated = [], []
    instance.cos, instance.sin = object(), object()
    return instance


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("grid_as_list", (True, False))
@pytest.mark.parametrize("align", (True, False))
@pytest.mark.parametrize("vectorized", (True, False))
def test_npu_graph_preprocessing_uses_exact_eager_method(
    monkeypatch, patch_module, dtype, grid_as_list, align, vectorized
):
    vision_cls, _ = _install_fake_upstream(monkeypatch)
    original_eager = vision_cls.forward
    patch_module.patch_qwen3_vl_npu_graph()
    instance = _vision(vision_cls, dtype=dtype, align=align, vectorized=vectorized)
    grid = [[1, 2, 4], [2, 2, 2]]
    raw = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 8
    supplied_grid = grid if grid_as_list else torch.tensor(grid)

    prepared, cu_seqlens, cos, sin = instance._prepare_graph_inputs(raw, supplied_grid)

    expected = raw.to(dtype) * 2 + instance._positions(grid, 2 if vectorized else 1)
    assert torch.equal(prepared, expected)
    assert prepared.dtype == dtype
    assert torch.equal(cu_seqlens, torch.tensor([0, 8, 12, 16], dtype=torch.int32))
    assert cos is instance.cos and sin is instance.sin
    assert instance.calls == [
        "patch_embed",
        ("vectorization_decision", 2),
        "vectorized" if vectorized else "from_list",
        ("rotary", grid),
    ]
    assert instance.align_corners is align
    assert instance.delegated == []
    assert vision_cls.forward is original_eager


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_non_npu_graph_preprocessing_delegates_unchanged(
    monkeypatch, patch_module, device
):
    vision_cls, _ = _install_fake_upstream(monkeypatch)
    patch_module.patch_qwen3_vl_npu_graph()
    instance = _vision(vision_cls)
    instance.device = torch.device(device)
    value, grid = object(), object()
    assert instance._prepare_graph_inputs(value, grid) == "original-device-path"
    assert instance.delegated == [(value, grid)]
    assert instance.calls == []


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("reorder", (False, True))
def test_replay_owns_output_and_forwards_all_original_arguments(
    monkeypatch, patch_module, dtype, reorder
):
    _, runner_cls = _install_fake_upstream(monkeypatch)
    patch_module.patch_qwen3_vl_npu_graph()
    runner = runner_cls()
    runner.static = torch.zeros(4, 3, dtype=dtype)
    runner.calls = []
    original = torch.arange(12, dtype=dtype).reshape(4, 1, 3)
    cos, sin = object(), object()
    indices = torch.tensor([3, 1]) if reorder else None
    expected = original.reshape(4, 3)
    if indices is not None:
        expected = expected.index_select(0, indices)

    first = runner.replay(4, original, cos, sin, indices)
    assert torch.equal(first, expected)
    assert first.dtype == dtype and first.shape == expected.shape
    assert (
        first.untyped_storage().data_ptr() != runner.static.untyped_storage().data_ptr()
    )
    assert runner.calls[0][0] == 4 and runner.calls[0][1] is original
    assert runner.calls[0][2] is cos and runner.calls[0][3] is sin
    assert runner.calls[0][4] is indices

    second = runner.replay(4, original + 17, cos, sin, indices)
    assert torch.equal(first, expected), (
        "later replay corrupted an earlier image's embeddings"
    )
    assert torch.equal(second, expected + 17)
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()


def test_original_static_output_reproduces_cross_replay_alias(monkeypatch):
    _, runner_cls = _install_fake_upstream(monkeypatch)
    runner = runner_cls()
    runner.static, runner.calls = torch.zeros(2, 3), []
    first = runner.replay(2, torch.ones(2, 1, 3))
    runner.replay(2, torch.full((2, 1, 3), 9.0))
    assert torch.equal(first, torch.full((2, 3), 9.0))


def test_patch_is_idempotent_and_preserves_both_signatures(monkeypatch, patch_module):
    vision_cls, runner_cls = _install_fake_upstream(monkeypatch)
    prepare, replay = vision_cls._prepare_graph_inputs, runner_cls.replay
    patch_module.patch_qwen3_vl_npu_graph()
    patched_prepare, patched_replay = (
        vision_cls._prepare_graph_inputs,
        runner_cls.replay,
    )
    patch_module.patch_qwen3_vl_npu_graph()
    assert vision_cls._prepare_graph_inputs is patched_prepare
    assert runner_cls.replay is patched_replay
    assert (
        patched_prepare.__wrapped__ is prepare and patched_replay.__wrapped__ is replay
    )
    assert inspect.signature(patched_prepare) == inspect.signature(prepare)
    assert inspect.signature(patched_replay) == inspect.signature(replay)


@pytest.mark.parametrize("bad_prepare,bad_replay", ((True, False), (False, True)))
def test_incompatible_interface_fails_transactionally(
    monkeypatch, patch_module, bad_prepare, bad_replay
):
    vision_cls, runner_cls = _install_fake_upstream(
        monkeypatch, bad_prepare=bad_prepare, bad_replay=bad_replay
    )
    prepare, replay = vision_cls._prepare_graph_inputs, runner_cls.replay
    with pytest.raises(RuntimeError, match="Unsupported SGLang interface"):
        patch_module.patch_qwen3_vl_npu_graph()
    assert vision_cls._prepare_graph_inputs is prepare
    assert runner_cls.replay is replay


@pytest.mark.parametrize("marked_target", ("prepare", "replay"))
def test_partial_patch_is_rejected_without_mutation(
    monkeypatch, patch_module, marked_target
):
    vision_cls, runner_cls = _install_fake_upstream(monkeypatch)
    prepare, replay = vision_cls._prepare_graph_inputs, runner_cls.replay
    setattr(
        prepare if marked_target == "prepare" else replay,
        patch_module._PATCH_MARKER,
        True,
    )
    with pytest.raises(RuntimeError, match="Incomplete Ascend ViT graph parity patch"):
        patch_module.patch_qwen3_vl_npu_graph()
    assert vision_cls._prepare_graph_inputs is prepare
    assert runner_cls.replay is replay
