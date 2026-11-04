# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CoreX compatibility patches must preserve dtype and import-order contracts."""

from dataclasses import dataclass
from contextlib import contextmanager
import builtins
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.iluvatar.patches import (
    clamp_position,
    conv_state_dtype,
    legacy_gpu_gate,
    speculative_tree,
)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _pdl_probe(out, ENABLE_PDL: tl.constexpr):
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()
        tl.store(out, 7)
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()


def test_clamp_uses_native_implementation_and_can_retry(monkeypatch):
    monkeypatch.setattr(clamp_position, "_applied", False)
    module = SimpleNamespace()
    monkeypatch.setattr(clamp_position.importlib, "import_module", lambda name: module)
    with pytest.raises(AttributeError):
        clamp_position.patch_clamp_position()
    assert not clamp_position._applied

    module._clamp_position_native = lambda lengths: (lengths - 1).clamp(min=0)
    clamp_position.patch_clamp_position()
    assert torch.equal(
        module.clamp_position(torch.tensor([0, 1, 8])), torch.tensor([0, 0, 7])
    )
    assert clamp_position._applied


@pytest.mark.parametrize("supported", [False, True])
def test_bf16_follows_vendor_support_instead_of_sm71(monkeypatch, supported):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 1))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: supported)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_context=lambda: SimpleNamespace(
                override=lambda *a, **kw: calls.append((a, kw))
            )
        ),
    )
    config = SimpleNamespace(dtype=torch.bfloat16)
    legacy_gpu_gate._downgrade_for_iluvatar(server_args=object(), model_config=config)
    assert config.dtype == (torch.bfloat16 if supported else torch.float16)
    assert len(calls) == (0 if supported else 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_explicit_non_bf16_dtype_is_preserved(monkeypatch, dtype):
    def unexpected_query(**kwargs):
        raise AssertionError("non-BF16 dtype must not query CUDA capabilities")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected_query)
    config = SimpleNamespace(dtype=dtype)
    legacy_gpu_gate._downgrade_for_iluvatar(model_config=config)
    assert config.dtype == dtype


def test_older_vendor_bf16_query_without_keyword(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    config = SimpleNamespace(dtype=torch.bfloat16)
    legacy_gpu_gate._downgrade_for_iluvatar(model_config=config)
    assert config.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "dynamic_cuda", [False, True], ids=["native-module", "vendor-proxy"]
)
def test_missing_bf16_query_requires_explicit_dtype(monkeypatch, dynamic_cuda):
    if dynamic_cuda:
        cuda = ModuleType("_vendor_cuda_proxy")

        def forwarded_attribute(name):
            if name == "is_bf16_supported":
                return lambda **kwargs: True
            raise AttributeError(name)

        cuda.__getattr__ = forwarded_attribute
        assert cuda.is_bf16_supported()
        assert "is_bf16_supported" not in vars(cuda)
        monkeypatch.setattr(torch, "cuda", cuda)
    # Deleting a concrete name can expose a vendor proxy's dynamic fallback.
    # An explicit None models a genuinely unavailable query on both runtimes.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", None)
    with pytest.raises(RuntimeError, match="select --dtype float16"):
        legacy_gpu_gate._downgrade_for_iluvatar(
            model_config=SimpleNamespace(dtype=torch.bfloat16)
        )


def test_legacy_gate_refreshes_copied_helper_and_can_retry(monkeypatch):
    monkeypatch.setattr(legacy_gpu_gate, "_patched", False)
    module = SimpleNamespace()
    monkeypatch.setattr(legacy_gpu_gate.importlib, "import_module", lambda name: module)
    with pytest.raises(AttributeError):
        legacy_gpu_gate.patch_legacy_gpu_gate()
    assert not legacy_gpu_gate._patched

    def original(**kw):
        return None

    module.maybe_downgrade_dtype_for_legacy_gpu = original
    runner = SimpleNamespace(maybe_downgrade_dtype_for_legacy_gpu=original)
    monkeypatch.setitem(sys.modules, legacy_gpu_gate._MODEL_RUNNER_MODULE, runner)
    legacy_gpu_gate.patch_legacy_gpu_gate()
    assert (
        module.maybe_downgrade_dtype_for_legacy_gpu
        is legacy_gpu_gate._downgrade_for_iluvatar
    )
    assert (
        runner.maybe_downgrade_dtype_for_legacy_gpu
        is legacy_gpu_gate._downgrade_for_iluvatar
    )


def _load_pdl_with_fake_triton(monkeypatch, cuda):
    triton, language = ModuleType("triton"), ModuleType("triton.language")
    triton.jit = lambda fn: fn
    language.extra = SimpleNamespace(cuda=cuda)
    language.static_assert = lambda condition, message: (
        None if condition else (_ for _ in ()).throw(RuntimeError(message))
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(legacy_gpu_gate.__file__).with_name("triton_pdl.py")
    spec = importlib.util.spec_from_file_location("_iluvatar_test_pdl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdl_is_false_missing_names_fail_and_real_intrinsic_is_preserved(monkeypatch):
    real_wait = object()
    cuda = SimpleNamespace(gdc_wait=real_wait)
    pdl = _load_pdl_with_fake_triton(monkeypatch, cuda)

    def original():
        return True

    utils = SimpleNamespace(is_arch_support_pdl=original)
    consumer = SimpleNamespace(is_arch_support_pdl=original)
    monkeypatch.setitem(sys.modules, "sgl_kernel.utils", utils)
    monkeypatch.setitem(sys.modules, "sglang._iluvatar_test_consumer", consumer)

    pdl.patch_triton_pdl_intrinsics()
    pdl.patch_triton_pdl_intrinsics()
    assert utils.is_arch_support_pdl() is False
    assert consumer.is_arch_support_pdl() is False
    assert cuda.gdc_wait is real_wait
    with pytest.raises(RuntimeError, match="not supported on Iluvatar CoreX"):
        cuda.gdc_launch_dependents()


def test_pdl_missing_package_does_not_mark_patch_complete(monkeypatch):
    pdl = _load_pdl_with_fake_triton(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        pdl.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing kernel shim")),
    )
    with pytest.raises(ImportError, match="missing kernel shim"):
        pdl.patch_triton_pdl_intrinsics()
    assert not pdl._applied


def test_overlay_runs_pdl_before_sglang_imports_and_retries(monkeypatch):
    calls = []
    fail = [True]

    def clamp():
        calls.append("clamp")
        if fail[0]:
            raise RuntimeError("transient import failure")

    prefix = "sglang_fl.dispatch.backends.vendor.iluvatar.patches"
    monkeypatch.setitem(
        sys.modules,
        prefix + ".triton_pdl",
        SimpleNamespace(patch_triton_pdl_intrinsics=lambda: calls.append("pdl")),
    )
    monkeypatch.setitem(
        sys.modules,
        prefix + ".speculative_tree",
        SimpleNamespace(patch_speculative_tree=lambda: calls.append("speculative")),
    )
    monkeypatch.setitem(
        sys.modules,
        prefix + ".breakable_stream_capture",
        SimpleNamespace(
            patch_breakable_stream_capture=lambda: calls.append("stream_capture")
        ),
    )
    monkeypatch.setattr(clamp_position, "patch_clamp_position", clamp)
    monkeypatch.setattr(
        legacy_gpu_gate,
        "patch_legacy_gpu_gate",
        lambda: calls.append("legacy"),
    )
    path = Path(legacy_gpu_gate.__file__).parents[1] / "patch.py"
    spec = importlib.util.spec_from_file_location(
        "sglang_fl.dispatch.backends.vendor.iluvatar._test_overlay",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="transient import failure"):
        spec.loader.exec_module(module)
    assert not module._patches_applied
    fail[0] = False
    module.apply_iluvatar_patches()
    module.apply_iluvatar_patches()
    assert calls == [
        "pdl",
        "clamp",
        "pdl",
        "clamp",
        "legacy",
        "stream_capture",
        "speculative",
    ]


def _load_model_config_bootstrap(monkeypatch, original, loader, active, hooks=None):
    import sglang_fl

    class ModelConfig:
        from_server_args = staticmethod(original)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.model_config",
        SimpleNamespace(ModelConfig=ModelConfig),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.plugins", SimpleNamespace(load_plugins=loader)
    )
    if hooks is None:
        hooks = SimpleNamespace(
            _hooks={
                "owned.target": [
                    (None, lambda: None, SimpleNamespace(plugin_name="sglang_fl"))
                ]
            },
            _patched={"owned.target"},
        )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(HookRegistry=hooks),
    )
    monkeypatch.setattr(sglang_fl, "is_plugin_active", lambda: active[0])
    path = Path(legacy_gpu_gate.__file__).parents[1] / "register_platform.py"
    spec = importlib.util.spec_from_file_location(
        "sglang_fl.dispatch.backends.vendor.iluvatar._test_register_platform", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return ModelConfig, module


def test_low_level_model_config_loads_plugin_before_copied_dtype_gate(monkeypatch):
    calls, active = [], [False]
    config = SimpleNamespace(dtype=torch.bfloat16)
    server_args = object()

    def nvidia_gate(**kwargs):
        raise RuntimeError("SGLang only supports sm75 and above.")

    helper = SimpleNamespace(maybe_downgrade_dtype_for_legacy_gpu=nvidia_gate)
    runner = SimpleNamespace(maybe_downgrade_dtype_for_legacy_gpu=nvidia_gate)
    monkeypatch.setitem(sys.modules, legacy_gpu_gate._TARGET_MODULE, helper)
    monkeypatch.setitem(sys.modules, legacy_gpu_gate._MODEL_RUNNER_MODULE, runner)
    monkeypatch.setattr(legacy_gpu_gate, "_patched", False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: True)

    def loader():
        calls.append("load_plugins")
        legacy_gpu_gate.patch_legacy_gpu_gate()
        calls.append("apply_hooks")
        active[0] = True

    def original(args, *, is_draft_model=False):
        calls.append("model_config")
        assert args is server_args and is_draft_model is True
        runner.maybe_downgrade_dtype_for_legacy_gpu(
            server_args=args, model_config=config
        )
        return config

    cls, module = _load_model_config_bootstrap(monkeypatch, original, loader, active)
    assert calls == []  # Installation during ModelRunner import must be deferred.
    wrapped = cls.__dict__["from_server_args"]
    assert isinstance(wrapped, staticmethod)
    module._install_model_config_bootstrap()
    assert cls.__dict__["from_server_args"] is wrapped
    for _ in range(2):
        assert cls.from_server_args(server_args, is_draft_model=True) is config
        assert config.dtype is torch.bfloat16
    assert calls == ["load_plugins", "apply_hooks", "model_config", "model_config"]


def test_active_engine_model_config_preserves_arguments_without_reentry(monkeypatch):
    calls, active = [], [True]
    arguments, result = object(), object()

    def unexpected_loader():
        raise AssertionError("An active Engine must not reenter plugin loading")

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    cls, _ = _load_model_config_bootstrap(
        monkeypatch, original, unexpected_loader, active
    )
    assert cls.from_server_args(arguments, model_path="draft") is result
    assert calls == [((arguments,), {"model_path": "draft"})]


@pytest.mark.parametrize("failure", ["inactive", "import", "hook", "swallowed_hook"])
def test_low_level_model_config_rejects_incomplete_plugin(monkeypatch, failure):
    calls, active = [], [False]

    def original(*args, **kwargs):
        calls.append("model_config")

    def loader():
        calls.append("load_plugins")
        if failure == "import":
            raise ImportError("missing runtime dependency")
        if failure == "hook":
            active[0] = True
            raise RuntimeError("hook application failed")
        if failure == "swallowed_hook":
            active[0] = True
        # Includes filtered plugins, or exceptions swallowed by SGLang's loader.

    hooks = SimpleNamespace(
        _hooks={
            "owned.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="sglang_fl"))
            ]
        },
        _patched=set(),
    )
    cls, _ = _load_model_config_bootstrap(monkeypatch, original, loader, active, hooks)
    error = ImportError if failure == "import" else RuntimeError
    with pytest.raises(error):
        cls.from_server_args(object())
    with pytest.raises(RuntimeError, match="previously failed"):
        cls.from_server_args(object())
    assert calls == ["load_plugins"]


def test_low_level_model_config_ignores_unrelated_plugin_hook_failure(monkeypatch):
    hooks = SimpleNamespace(
        _hooks={
            "owned.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="sglang_fl"))
            ],
            "other.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="another_plugin"))
            ],
        },
        _patched={"owned.target"},
    )
    result = object()
    cls, _ = _load_model_config_bootstrap(
        monkeypatch, lambda value: result, lambda: None, [True], hooks
    )
    assert cls.from_server_args(object()) is result


@pytest.mark.parametrize("reset", ["active", "hooks"])
def test_low_level_model_config_rejects_reset_after_initialization(monkeypatch, reset):
    active, calls = [True], []
    hooks = SimpleNamespace(
        _hooks={
            "owned.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="sglang_fl"))
            ]
        },
        _patched={"owned.target"},
    )
    cls, _ = _load_model_config_bootstrap(
        monkeypatch, lambda value: calls.append(value), lambda: None, active, hooks
    )
    cls.from_server_args("before")
    if reset == "active":
        active[0] = False
    else:
        hooks._hooks.clear()
        hooks._patched.clear()
    with pytest.raises(RuntimeError):
        cls.from_server_args("after")
    assert calls == ["before"]


def test_low_level_model_config_waits_for_hooks_after_active_flag(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    active, calls = [False], []
    active_set, finish_hooks = threading.Event(), threading.Event()
    second_entered, second_returned = threading.Event(), threading.Event()

    def loader():
        calls.append("load_plugins")
        active[0] = True
        active_set.set()
        assert finish_hooks.wait(5)
        calls.append("apply_hooks")

    def original(value):
        assert "apply_hooks" in calls
        calls.append(value)
        return value

    cls, _ = _load_model_config_bootstrap(monkeypatch, original, loader, active)

    def second_call():
        second_entered.set()
        value = cls.from_server_args("second")
        second_returned.set()
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cls.from_server_args, "first")
        second = None
        try:
            assert active_set.wait(5)
            second = pool.submit(second_call)
            assert second_entered.wait(5)
            assert not second_returned.wait(0.1)
        finally:
            finish_hooks.set()
        assert first.result(timeout=5) == "first"
        assert second.result(timeout=5) == "second"
    assert calls[:2] == ["load_plugins", "apply_hooks"]
    assert sorted(calls[2:]) == ["first", "second"]


def test_low_level_model_config_rejects_recursive_bootstrap(monkeypatch):
    active, calls = [False], []

    def loader():
        active[0] = True
        cls.from_server_args("recursive")

    cls, _ = _load_model_config_bootstrap(
        monkeypatch, lambda value: calls.append(value), loader, active
    )
    with pytest.raises(RuntimeError, match="Recursive"):
        cls.from_server_args("outer")
    with pytest.raises(RuntimeError, match="previously failed"):
        cls.from_server_args("later")
    assert calls == []


def test_low_level_model_config_preserves_plugin_opt_out_environment(monkeypatch):
    import os

    active = [False]
    expected = {
        "USE_FLAGGEMS": "0",
        "SGLANG_FL_OOT_ENABLED": "0",
        "SGLANG_PLUGINS": "sglang_fl",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    def loader():
        assert {key: os.environ[key] for key in expected} == expected
        active[0] = True

    result = object()
    cls, _ = _load_model_config_bootstrap(
        monkeypatch, lambda args: result, loader, active
    )
    assert cls.from_server_args(object()) is result
    assert {key: os.environ[key] for key in expected} == expected


def _cpu_build_speculative_tree(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode=0,
):
    """Small graph oracle for exercising the bridge's output-buffer contract."""
    if tree_mask_mode == 2:
        raise NotImplementedError("QLEN_ONLY_BITPACKING")
    offset = 0
    for batch, length in enumerate(verified_seq_len.tolist()):
        selected = selected_index[batch].tolist()
        parents = [-1]
        for selected_token in selected:
            block = selected_token // topk
            parents.append(
                0 if block == 0 else selected.index(int(parent_list[batch, block])) + 1
            )
        width = draft_token_num + (length if tree_mask_mode == 0 else 0)
        mask = tree_mask[offset : offset + draft_token_num * width].view(
            draft_token_num, width
        )
        for node in range(draft_token_num):
            retrieve_index[batch, node] = batch * draft_token_num + node
            path = []
            ancestor = node
            while ancestor != -1:
                path.append(ancestor)
                ancestor = parents[ancestor]
            assert len(path) - 1 <= depth
            positions[batch * draft_token_num + node] = length + len(path) - 1
            mask[node, -draft_token_num:] = False
            mask[node, [width - draft_token_num + index for index in path]] = True
            children = [
                index for index in range(1, draft_token_num) if parents[index] == node
            ]
            if children:
                retrieve_next_token[batch, node] = children[0]
                for left, right in zip(children, children[1:]):
                    retrieve_next_sibling[batch, left] = right
        offset += draft_token_num * width


def _cpu_verify_speculative_tree(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
):
    """Serial greedy tree walk, including the terminal bonus prediction."""
    for row in range(candidates.shape[0]):
        node = accepted = 0
        last = int(retrieve_index[row, node])
        accept_index[row, 0] = last
        for slot in range(1, accept_index.shape[1]):
            child = int(retrieve_next_token[row, node])
            predicted = int(target_predict.flatten()[last])
            while child != -1 and int(candidates[row, child]) != predicted:
                child = int(retrieve_next_sibling[row, child])
            if child == -1:
                break
            predicts.flatten()[last] = predicted
            last = int(retrieve_index[row, child])
            accept_index[row, slot] = last
            accepted += 1
            node = child
        predicts.flatten()[last] = target_predict.flatten()[last]
        accept_token_num[row] = accepted


@pytest.fixture
def speculative_module(monkeypatch):
    from sglang_fl import utils

    monkeypatch.setattr(speculative_tree, "_applied", False)
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name="iluvatar")
    )
    module = ModuleType(speculative_tree._MODULE)
    module.sgl_build_tree_kernel_triton = _cpu_build_speculative_tree
    module.verify_tree_greedy_triton = _cpu_verify_speculative_tree

    def unavailable(*args, **kwargs):
        pytest.fail("The NVIDIA import shim must not execute")

    def original_verify(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        topk=-1,
    ):
        unavailable()

    module.sgl_build_tree_kernel_efficient = unavailable
    module.verify_tree_greedy_func = original_verify
    spec_utils = ModuleType(speculative_tree._SPEC_MODULE)
    spec_utils.fast_topk = unavailable
    common = ModuleType("sglang.srt.utils.common")

    def fast_topk(values, topk, dim):
        return (
            torch.max(values, dim=dim, keepdim=True)
            if topk == 1
            else torch.topk(values, topk, dim=dim)
        )

    common.fast_topk = fast_topk
    for item in (module, spec_utils, common):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    return module, spec_utils


def _speculative_buffers(mode=0):
    lengths = torch.tensor([2, 3])
    size = 16 * 2 + (int(lengths.sum()) * 4 if mode == 0 else 0)
    return (
        torch.tensor([[0, 0], [0, 1]]),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        lengths,
        torch.ones(size, dtype=torch.bool),
        torch.full((8,), -1),
        torch.full((2, 4), -1),
        torch.full((2, 4), -1),
        torch.full((2, 4), -1),
    )


@pytest.mark.parametrize("mode", [0, 1], ids=["full-mask", "qlen-only"])
def test_speculative_build_retains_branch_positions_and_masks(speculative_module, mode):
    module, _ = speculative_module
    speculative_tree.patch_speculative_tree()
    buffers = _speculative_buffers(mode)
    module.sgl_build_tree_kernel_efficient(*buffers, 2, 2, 4, mode)
    torch.testing.assert_close(buffers[4], torch.tensor([2, 3, 3, 4, 3, 4, 4, 5]))
    torch.testing.assert_close(buffers[5], torch.arange(8).reshape(2, 4))
    torch.testing.assert_close(
        buffers[6], torch.tensor([[1, 3, -1, -1], [1, -1, 3, -1]])
    )
    torch.testing.assert_close(
        buffers[7], torch.tensor([[-1, 2, -1, -1], [-1, 2, -1, -1]])
    )
    ancestry = [
        torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.bool
        ),
        torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]], dtype=torch.bool
        ),
    ]
    prefixes = [2, 3] if mode == 0 else [0, 0]
    expected = torch.cat(
        [
            torch.cat((torch.ones(4, prefix, dtype=torch.bool), mask), dim=1).flatten()
            for prefix, mask in zip(prefixes, ancestry)
        ]
    )
    torch.testing.assert_close(buffers[3], expected)


@pytest.mark.parametrize(
    "root_match", [False, True], ids=["root-rejected", "full-branch"]
)
def test_speculative_verify_updates_acceptance_and_bonus_in_place(
    speculative_module, root_match
):
    module, _ = speculative_module
    speculative_tree.patch_speculative_tree()
    buffers = _speculative_buffers()
    module.sgl_build_tree_kernel_efficient(*buffers, 2, 2, 4, 0)
    predicts = torch.full((8,), -7)
    accepted = torch.full((2, 3), -1)
    count = torch.full((2,), -1)
    candidates = torch.tensor([[100, 11, 22, 33], [200, 44, 55, 66]])
    targets = torch.tensor([[11 if root_match else 99, 33, 90, 777], [55, 80, 123, 70]])
    result = module.verify_tree_greedy_func(
        predicts, accepted, count, candidates, *buffers[5:], targets, topk=7
    )
    assert all(
        actual is expected
        for actual, expected in zip(result, (predicts, accepted, count))
    )
    torch.testing.assert_close(
        accepted, torch.tensor([[0, 1, 3] if root_match else [0, -1, -1], [4, 6, -1]])
    )
    torch.testing.assert_close(count, torch.tensor([2 if root_match else 0, 1]))
    expected = (
        torch.tensor([11, 33, -7, 777, 55, -7, 123, -7])
        if root_match
        else torch.tensor([99, -7, -7, -7, 55, -7, 123, -7])
    )
    torch.testing.assert_close(predicts, expected)


@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_speculative_topk_preserves_values_indices_and_shapes(
    speculative_module, topk, dtype
):
    _, spec_utils = speculative_module
    speculative_tree.patch_speculative_tree()
    values, indices = spec_utils.fast_topk(
        torch.tensor([[0.2, 0.8, 0.4], [0.9, 0.1, 0.3]], dtype=dtype), topk, dim=-1
    )
    expected_values = [[0.8], [0.9]] if topk == 1 else [[0.8, 0.4], [0.9, 0.3]]
    expected_indices = [[1], [0]] if topk == 1 else [[1, 2], [0, 2]]
    torch.testing.assert_close(values, torch.tensor(expected_values, dtype=dtype))
    torch.testing.assert_close(indices, torch.tensor(expected_indices))


def test_speculative_patch_retries_and_preserves_official_rejection(
    monkeypatch, speculative_module
):
    module, _ = speculative_module
    original_build = module.sgl_build_tree_kernel_efficient
    module.verify_tree_greedy_triton = lambda incompatible: None
    with pytest.raises(RuntimeError, match="0.5.18"):
        speculative_tree.patch_speculative_tree()
    assert not speculative_tree._applied
    assert module.sgl_build_tree_kernel_efficient is original_build
    module.verify_tree_greedy_triton = _cpu_verify_speculative_tree
    speculative_tree.patch_speculative_tree()
    original_verify = module.verify_tree_greedy_func
    speculative_tree.patch_speculative_tree()
    assert module.verify_tree_greedy_func is original_verify
    buffers = _speculative_buffers()
    with pytest.raises(NotImplementedError, match="BITPACKING"):
        module.sgl_build_tree_kernel_efficient(*buffers, 2, 2, 4, 2)
    assert torch.equal(buffers[4], torch.full((8,), -1))


@pytest.mark.parametrize("vendor", [None, "nvidia", "mthreads", "ascend", "hygon"])
def test_speculative_patch_does_not_import_or_mutate_other_vendors(
    monkeypatch, speculative_module, vendor
):
    from sglang_fl import utils

    module, spec_utils = speculative_module
    before = module.verify_tree_greedy_func, spec_utils.fast_topk
    monkeypatch.setattr(
        utils,
        "get_device_info",
        lambda: None if vendor is None else SimpleNamespace(vendor_name=vendor),
    )
    monkeypatch.setattr(
        speculative_tree.importlib,
        "import_module",
        lambda name: pytest.fail("Foreign vendor must not import speculative kernels"),
    )
    speculative_tree.patch_speculative_tree()
    assert not speculative_tree._applied
    assert before == (module.verify_tree_greedy_func, spec_utils.fast_topk)


def test_speculative_patch_refreshes_existing_copies_without_changing_shim(
    monkeypatch, speculative_module
):
    module, spec_utils = speculative_module
    old_build, old_verify, old_topk = (
        module.sgl_build_tree_kernel_efficient,
        module.verify_tree_greedy_func,
        spec_utils.fast_topk,
    )
    consumer = SimpleNamespace(
        sgl_build_tree_kernel_efficient=old_build,
        verify_tree_greedy_func=old_verify,
        fast_topk=old_topk,
    )
    shim = SimpleNamespace(
        build_tree_kernel_efficient=old_build,
        verify_tree_greedy=old_verify,
        fast_topk=old_topk,
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative._copied_test", consumer)
    monkeypatch.setitem(sys.modules, "sgl_kernel", shim)
    speculative_tree.patch_speculative_tree()
    assert consumer.sgl_build_tree_kernel_efficient is _cpu_build_speculative_tree
    assert consumer.verify_tree_greedy_func is module.verify_tree_greedy_func
    assert consumer.fast_topk is spec_utils.fast_topk
    assert (
        shim.build_tree_kernel_efficient,
        shim.verify_tree_greedy,
        shim.fast_topk,
    ) == (old_build, old_verify, old_topk)


@pytest.mark.gpu
def test_corex_dead_pdl_branch_compiles_and_live_pdl_is_rejected(monkeypatch):
    if triton is None or not hasattr(torch, "corex") or not torch.cuda.is_available():
        pytest.skip("CoreX hardware and vendor Triton are required")
    pdl = importlib.import_module(
        "sglang_fl.dispatch.backends.vendor.iluvatar.patches.triton_pdl"
    )
    monkeypatch.setattr(pdl, "_applied", False)
    monkeypatch.setattr(tl.extra, "cuda", SimpleNamespace(), raising=False)
    utils = importlib.import_module("sgl_kernel.utils")
    monkeypatch.setattr(utils, "is_arch_support_pdl", lambda: True)
    pdl.patch_triton_pdl_intrinsics()
    assert utils.is_arch_support_pdl() is False
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    _pdl_probe[(1,)](output, False)
    assert output.item() == 7
    with pytest.raises(triton.CompilationError) as error:
        _pdl_probe[(1,)](output, True)
    causes, cause = [], error.value
    while cause is not None:
        causes.append(str(cause))
        cause = cause.__cause__
    assert "CUDA PDL is not supported on Iluvatar CoreX" in "\n".join(causes)


@pytest.fixture
def qwen_configs(monkeypatch):
    @dataclass(frozen=True)
    class StateDType:
        conv: object
        temporal: object

    @dataclass(frozen=True)
    class CacheParams:
        dtype: StateDType
        shape: object
        layers: object

    class Qwen3NextConfig:
        def __init__(self):
            self.original = CacheParams(
                StateDType(torch.bfloat16, torch.float32), object(), [0, 2]
            )

        @property
        def mamba2_cache_params(self):
            return self.original

    class Qwen3_5TextConfig(Qwen3NextConfig):
        pass

    class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
        pass

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.qwen3_next",
        SimpleNamespace(Qwen3NextConfig=Qwen3NextConfig),
    )
    monkeypatch.delenv("SGLANG_MAMBA_CONV_DTYPE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_context=lambda: SimpleNamespace(override=lambda *a, **k: None)
        ),
    )
    return Qwen3NextConfig, Qwen3_5TextConfig, Qwen3_5MoeTextConfig


@pytest.mark.parametrize("subclass", [1, 2], ids=["dense", "moe"])
@pytest.mark.parametrize(
    "requested,supported,effective",
    [
        (torch.float16, None, torch.float16),
        (torch.bfloat16, True, torch.bfloat16),
        (torch.bfloat16, False, torch.float16),
        (torch.float32, None, torch.float32),
    ],
)
def test_final_runner_dtype_controls_only_its_conv_history(
    monkeypatch, qwen_configs, subclass, requested, supported, effective
):
    def support_query(**kwargs):
        assert supported is not None, "Explicit FP16/FP32 must not query BF16 support"
        return supported

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", support_query)
    text_config = qwen_configs[subclass]()
    config = SimpleNamespace(dtype=requested, hf_text_config=text_config)
    original = text_config.mamba2_cache_params
    legacy_gpu_gate._downgrade_for_iluvatar(model_config=config)
    actual = text_config.mamba2_cache_params
    assert config.dtype == effective
    assert actual.dtype.conv == effective
    assert actual.dtype.temporal is original.dtype.temporal
    assert actual.shape is original.shape
    assert actual.layers is original.layers
    assert original.dtype.conv == torch.bfloat16
    assert isinstance(getattr(text_config, conv_state_dtype._MODEL_DTYPE), str)
    json.dumps(
        {
            conv_state_dtype._MODEL_DTYPE: getattr(
                text_config, conv_state_dtype._MODEL_DTYPE
            )
        }
    )


def test_float16_then_bfloat16_and_unmarked_configs_are_independent(
    monkeypatch, qwen_configs
):
    base, dense, moe = qwen_configs
    first, second, unmarked = dense(), moe(), base()
    original_unmarked = unmarked.mamba2_cache_params
    conv_state_dtype.record_model_conv_dtype(
        SimpleNamespace(dtype=torch.float16, hf_text_config=first)
    )
    getter = base.mamba2_cache_params.fget
    conv_state_dtype.record_model_conv_dtype(
        SimpleNamespace(dtype=torch.bfloat16, hf_text_config=second)
    )
    conv_state_dtype.record_model_conv_dtype(
        SimpleNamespace(dtype=torch.float16, hf_text_config=first)
    )
    assert base.mamba2_cache_params.fget is getter
    assert first.mamba2_cache_params.dtype.conv == torch.float16
    assert second.mamba2_cache_params.dtype.conv == torch.bfloat16
    assert unmarked.mamba2_cache_params is original_unmarked
    assert "SGLANG_MAMBA_CONV_DTYPE" not in __import__("os").environ


@pytest.mark.parametrize("explicit", ["bfloat16", "float16", "float32", ""])
def test_explicit_conv_state_precision_keeps_original_precedence(
    monkeypatch, qwen_configs, explicit
):
    config = qwen_configs[1]()
    original = config.mamba2_cache_params
    monkeypatch.setenv("SGLANG_MAMBA_CONV_DTYPE", explicit)
    conv_state_dtype.record_model_conv_dtype(
        SimpleNamespace(dtype=torch.float16, hf_text_config=config)
    )
    assert config.mamba2_cache_params is original


def test_other_configs_do_not_patch_qwen_or_gain_a_marker(qwen_configs):
    base = qwen_configs[0]
    original = base.mamba2_cache_params
    other = SimpleNamespace()
    conv_state_dtype.record_model_conv_dtype(
        SimpleNamespace(dtype=torch.float16, hf_text_config=other)
    )
    assert base.mamba2_cache_params is original
    assert not hasattr(other, conv_state_dtype._MODEL_DTYPE)


def test_incompatible_property_can_retry_without_a_partial_marker(
    monkeypatch, qwen_configs
):
    base, dense, _ = qwen_configs
    original = base.mamba2_cache_params
    config = dense()
    runner = SimpleNamespace(dtype=torch.float16, hf_text_config=config)
    monkeypatch.setattr(base, "mamba2_cache_params", object())
    with pytest.raises(RuntimeError, match="property"):
        conv_state_dtype.record_model_conv_dtype(runner)
    assert not hasattr(config, conv_state_dtype._MODEL_DTYPE)
    monkeypatch.setattr(base, "mamba2_cache_params", original)
    conv_state_dtype.record_model_conv_dtype(runner)
    assert config.mamba2_cache_params.dtype.conv == torch.float16


class _SideStreamFakeCuda:
    def __init__(self):
        self.current = "original-stream"
        self.capturing = False
        self.error = None
        self.events = []

    @contextmanager
    def stream(self, selected):
        previous = self.current
        self.events.append(("enter", selected, previous))
        self.current = selected
        try:
            yield
        finally:
            self.current = previous
            self.events.append(("restore", previous))

    def is_current_stream_capturing(self):
        self.events.append(("query", self.current))
        if self.error is not None:
            raise self.error
        return self.capturing


@pytest.fixture
def side_stream_context(monkeypatch, record_property):
    candidate = importlib.import_module(
        "sglang_fl.dispatch.backends.vendor.iluvatar.patches.breakable_stream_capture"
    )
    monkeypatch.setattr(candidate, "_applied", False)
    monkeypatch.setattr(candidate, "_original_is_stream_capturing", None)
    cuda = _SideStreamFakeCuda()
    torch = ModuleType("torch")
    torch.corex = object()
    torch.cuda = cuda
    torch.is_hip = object()
    torch.xpu = object()
    from sglang_fl import utils

    state = SimpleNamespace(info=SimpleNamespace(vendor_name="iluvatar"))
    monkeypatch.setattr(utils, "get_device_info", lambda: state.info)
    monkeypatch.setitem(sys.modules, "torch", torch)
    target = ModuleType(candidate._TARGET_MODULE)
    original = lambda stream: "original-result"
    target._is_stream_capturing = original
    imports = []

    def import_target(name):
        imports.append(name)
        assert name == candidate._TARGET_MODULE
        return target

    monkeypatch.setattr(
        candidate, "importlib", SimpleNamespace(import_module=import_target)
    )
    record_property("CPU_contract_only", True)
    record_property("GPU_executions", 0)
    record_property("GPU_runtime_executed", False)
    return SimpleNamespace(
        candidate=candidate,
        cuda=cuda,
        torch=torch,
        state=state,
        target=target,
        original=original,
        imports=imports,
    )


@pytest.mark.parametrize("answer", [False, True])
def test_corex_side_stream_requested_stream_query_and_return_restore(
    side_stream_context, answer
):
    x = side_stream_context
    x.cuda.capturing = answer
    original_hip, original_xpu = x.torch.is_hip, x.torch.xpu
    assert x.candidate.patch_breakable_stream_capture() is True
    assert x.target._is_stream_capturing is x.candidate._is_stream_capturing_corex
    assert x.target._is_stream_capturing("selected-stream") is answer
    assert x.cuda.events == [
        ("enter", "selected-stream", "original-stream"),
        ("query", "selected-stream"),
        ("restore", "original-stream"),
    ]
    assert x.cuda.current == "original-stream"
    assert (x.torch.is_hip, x.torch.xpu) == (original_hip, original_xpu)
    assert x.candidate._original_is_stream_capturing is x.original
    assert x.candidate._applied is True


def test_corex_side_stream_query_exception_restores_and_propagates_exact_exception(
    side_stream_context,
):
    x = side_stream_context
    failure = RuntimeError("actual query failed")
    x.cuda.error = failure
    x.candidate.patch_breakable_stream_capture()
    with pytest.raises(RuntimeError) as caught:
        x.target._is_stream_capturing("selected-stream")
    assert caught.value is failure
    assert x.cuda.current == "original-stream"
    assert x.cuda.events[-1] == ("restore", "original-stream")


@pytest.mark.parametrize("vendor", [None, "nvidia", "ascend", "hygon", "metax"])
def test_corex_side_stream_other_vendors_import_neither_target_nor_torch(
    side_stream_context, monkeypatch, vendor
):
    x = side_stream_context
    x.state.info = None if vendor is None else SimpleNamespace(vendor_name=vendor)
    actual_import = builtins.__import__

    def forbid_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(
                "Non-Iluvatar compatibility path must not import Torch"
            )
        return actual_import(name, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", forbid_torch)
        assert x.candidate.patch_breakable_stream_capture() is False
    assert x.imports == []
    assert x.target._is_stream_capturing is x.original
    assert x.candidate._original_is_stream_capturing is None
    assert x.candidate._applied is False


def test_corex_side_stream_iluvatar_without_corex_rejected_before_target_import(
    side_stream_context,
):
    x = side_stream_context
    del x.torch.corex
    with pytest.raises(RuntimeError, match="requires CoreX"):
        x.candidate.patch_breakable_stream_capture()
    assert x.imports == []
    assert x.target._is_stream_capturing is x.original
    assert x.candidate._original_is_stream_capturing is None
    assert x.candidate._applied is False


@pytest.mark.parametrize("missing", ["cuda", "stream", "is_current_stream_capturing"])
def test_corex_side_stream_missing_real_cuda_api_has_no_partial_patch(
    side_stream_context, missing
):
    x = side_stream_context
    if missing == "cuda":
        del x.torch.cuda
    else:
        setattr(x.cuda, missing, None)
    with pytest.raises(RuntimeError, match="APIs are required"):
        x.candidate.patch_breakable_stream_capture()
    assert x.imports == []
    assert x.target._is_stream_capturing is x.original
    assert x.candidate._original_is_stream_capturing is None
    assert x.candidate._applied is False


@pytest.mark.parametrize("invalid", [None, 7])
def test_corex_side_stream_missing_or_noncallable_target_has_no_partial_patch(
    side_stream_context, invalid
):
    x = side_stream_context
    if invalid is None:
        del x.target._is_stream_capturing
    else:
        x.target._is_stream_capturing = invalid
    with pytest.raises(RuntimeError, match="binding is missing"):
        x.candidate.patch_breakable_stream_capture()
    assert getattr(x.target, "_is_stream_capturing", None) is invalid
    assert x.candidate._original_is_stream_capturing is None
    assert x.candidate._applied is False


def test_corex_side_stream_repeated_call_preserves_original_and_replacement_identity(
    side_stream_context,
):
    x = side_stream_context
    assert x.candidate.patch_breakable_stream_capture() is True
    replacement = x.target._is_stream_capturing
    assert replacement is not x.original
    assert replacement is x.candidate._is_stream_capturing_corex
    assert x.candidate.patch_breakable_stream_capture() is True
    assert x.target._is_stream_capturing is replacement
    assert x.candidate._original_is_stream_capturing is x.original
    assert x.imports == [x.candidate._TARGET_MODULE]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("automatic_join", [False, True], ids=["explicit", "automatic"])
def test_corex_breakable_side_stream_capture_and_replay(dtype, automatic_join):
    """Real graph forks must query CoreX and safely join both sides of a break."""
    import gc
    import weakref

    if not hasattr(torch, "corex") or not torch.cuda.is_available():
        pytest.skip("CoreX hardware is required")
    from sglang_fl.dispatch.backends.vendor.iluvatar.patch import apply_iluvatar_patches

    apply_iluvatar_patches()
    bcg = importlib.import_module(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph."
        "breakable_cuda_graph"
    )
    stream_compat = importlib.import_module(
        "sglang_fl.dispatch.backends.vendor.iluvatar.patches.breakable_stream_capture"
    )
    assert bcg._is_stream_capturing is stream_compat._is_stream_capturing_corex
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    current = torch.cuda.current_stream().cuda_stream
    assert bcg._is_stream_capturing(side) is False
    assert torch.cuda.current_stream().cuda_stream == current
    source = torch.arange(16, dtype=torch.float32, device="cpu").to(dtype).cuda()
    torch.cuda.synchronize()
    main.wait_stream(torch.cuda.current_stream())
    tracked = {}

    @bcg.eager_on_graph(True)
    def bridge(x):
        return x + 1

    def forward(track=False):
        side.wait_stream(main)
        with torch.cuda.stream(side):
            intermediate = source * 2
        if track:
            before = torch.cuda.current_stream().cuda_stream
            assert bcg._is_stream_capturing(side) is True
            assert torch.cuda.current_stream().cuda_stream == before
            assert side in bcg._forked_streams_var.get()
            tracked["tensor"] = weakref.ref(intermediate)
            tracked["storage"] = intermediate.untyped_storage()._weak_ref()
        if not (track and automatic_join):
            main.wait_stream(side)
        if track:
            assert len(bcg._forked_streams_var.get()) == int(automatic_join)
        middle = bridge(intermediate)
        if track:
            assert not bcg._forked_streams_var.get()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            output = middle * 3
        if track:
            before = torch.cuda.current_stream().cuda_stream
            assert bcg._is_stream_capturing(side) is True
            assert torch.cuda.current_stream().cuda_stream == before
            assert side in bcg._forked_streams_var.get()
            tracked["forks"] = bcg._forked_streams_var.get()
        if not (track and automatic_join):
            main.wait_stream(side)
        if track:
            assert len(bcg._forked_streams_var.get()) == int(automatic_join)
        return output

    with torch.cuda.stream(main):
        for _ in range(3):
            warm = forward()
    main.synchronize()
    del warm
    graph, pool = bcg.BreakableCUDAGraph(), torch.cuda.graph_pool_handle()
    try:
        with bcg.BreakableCUDAGraphCapture(graph, pool=pool, stream=main):
            output = forward(track=True)
        torch.cuda.synchronize()
        gc.collect()
        assert len(graph._segments) == 2 and len(graph._break_fns) == 1
        assert not tracked["forks"]
        assert tracked["tensor"]() is None
        assert torch.UntypedStorage._expired(tracked["storage"])
        assert bcg._is_stream_capturing(side) is False
        for delta in (3, 5, 7):
            source.copy_(
                torch.tensor([i + delta for i in range(16)], dtype=dtype, device="cpu")
            )
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            actual = output.detach().cpu()
            expected = torch.tensor(
                [(2 * (i + delta) + 1) * 3 for i in range(16)],
                dtype=dtype,
                device="cpu",
            )
            assert torch.isfinite(actual).all()
            assert torch.equal(actual, expected)
    finally:
        if "storage" in tracked:
            torch.UntypedStorage._free_weak_ref(tracked["storage"])
