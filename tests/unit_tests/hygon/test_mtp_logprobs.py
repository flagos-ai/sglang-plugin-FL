# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for HCU's speculative-logprob padding equivalence and hooks."""

import builtins
from functools import partial
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
from threading import Barrier, Thread
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor/hygon/patches/mtp_logprobs.py"
)
TARGET = "sglang.srt.layers.logprob_processor"
COMMON = "sglang.srt.speculative.eagle_worker_common"


@pytest.fixture
def guard(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "isolated_hcu_mtp_logprob_guard", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch, "__hcu_version__", "3.6", raising=False)
    for name in (TARGET, COMMON):
        monkeypatch.delitem(sys.modules, name, raising=False)
    processor = ModuleType(TARGET)
    processor.normalizer_inputs = []
    processor.top_requests = []
    processor.token_requests = []
    processor.return_original = False
    processor.envs = SimpleNamespace(
        SGLANG_RETURN_ORIGINAL_LOGPROB=SimpleNamespace(
            get=lambda: processor.return_original
        )
    )

    def row_normalizer(logits):
        processor.normalizer_inputs.append(logits.detach().clone())
        maximum = logits.float().amax(dim=-1)
        return maximum, torch.logsumexp(logits.float() - maximum[:, None], dim=-1)

    def top(logprobs, nums, no_copy_to_cpu=False):
        assert no_copy_to_cpu
        processor.top_requests.append(nums)
        values, ids = logprobs.topk(max(nums), dim=-1)
        return [values[i, :n] for i, n in enumerate(nums)], [
            ids[i, :n] for i, n in enumerate(nums)
        ]

    def tokens(logprobs, ids, no_copy_to_cpu=False):
        assert no_copy_to_cpu
        processor.token_requests.append(ids)
        return [
            logprobs[i, row] if row is not None else [] for i, row in enumerate(ids)
        ], [row or [] for row in ids]

    processor.compute_row_log_normalizer = row_normalizer
    processor.get_top_logprobs = top
    processor.get_token_ids_logprobs = tokens
    monkeypatch.setitem(sys.modules, TARGET, processor)
    module._test_processor = processor
    yield module


def inputs():
    batch = SimpleNamespace(
        seq_lens=torch.tensor([27, 25, 40, 28]),
        sampling_info=SimpleNamespace(
            is_all_greedy=True, temperatures=torch.ones(4, 1)
        ),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits = SimpleNamespace(
        next_token_logits=torch.arange(112, dtype=torch.float32).reshape(16, 7)
    )
    predict = torch.tensor(
        [2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 1, 1, 2, 0, 0], dtype=torch.int32
    )
    index = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, -1, -1]],
        dtype=torch.int32,
    )
    return batch, logits, predict, index


def spy():
    calls = []

    def original(
        batch, logits_output, predict, *, accept_index=None, chain_stride=None
    ):
        calls.append((batch, logits_output, predict, accept_index, chain_stride))
        return calls[-1]

    return original, calls


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_actual_padding_maps_to_last_row_without_mutating_original(guard, dtype):
    batch, logits, predict, index = inputs()
    index = index.to(dtype)
    before = index.clone()
    original, calls = spy()
    result = guard._compute_spec_logprobs_hook(
        original, batch, logits, predict, accept_index=index
    )
    normalized = guard.normalize_hcu_spec_logprob_accept_index(index, 16)
    assert result is None and not calls
    assert normalized.tolist()[-1] == [12, 13, 15, 15]
    assert normalized.dtype == index.dtype and torch.equal(index, before)
    assert torch.equal(predict[normalized.long()], predict[index.long()])
    assert torch.equal(
        guard._test_processor.normalizer_inputs[0],
        logits.next_token_logits[index.long().flatten()],
    )
    reference = torch.nn.functional.log_softmax(
        logits.next_token_logits[index.long().flatten()].double(), dim=-1
    )
    expected = reference.gather(
        1, predict[index.long().flatten()].long()[:, None]
    ).view(4, 4)
    torch.testing.assert_close(
        logits.next_token_logprobs.double(), expected, rtol=1e-5, atol=2e-5
    )
    assert not torch.equal(
        predict[index.long().clamp(min=0)], predict[normalized.long()]
    )


@pytest.mark.parametrize("bad", [-2, -16, 16, 999])
def test_invalid_indices_fail_before_upstream_call(guard, bad):
    batch, logits, predict, index = inputs()
    index[-1, -1] = bad
    original, calls = spy()
    with pytest.raises(ValueError, match="must lie"):
        partial(guard._compute_spec_logprobs_hook, original)(
            batch, logits, predict, accept_index=index
        )
    assert not calls


@pytest.mark.parametrize(
    "index", [torch.tensor([0]), torch.tensor([[0.0]]), torch.tensor([[True]])]
)
def test_helper_rejects_rank_or_noninteger_indices(guard, index):
    with pytest.raises(ValueError, match="2D integer"):
        guard.normalize_hcu_spec_logprob_accept_index(index, 16)


def test_no_padding_and_empty_inputs_keep_original_index(guard):
    index = torch.tensor([[0, 1], [2, 3]])
    assert guard.normalize_hcu_spec_logprob_accept_index(index, 4) is index
    empty = torch.empty((0, 4), dtype=torch.int32)
    assert guard.normalize_hcu_spec_logprob_accept_index(empty, 0) is empty
    original, calls = spy()
    partial(guard._compute_spec_logprobs_hook, original)(
        SimpleNamespace(
            seq_lens=[],
            sampling_info=SimpleNamespace(is_all_greedy=True),
            top_logprobs_nums=None,
            token_ids_logprobs=None,
        ),
        SimpleNamespace(next_token_logits=torch.empty(0, 7)),
        torch.empty(0, dtype=torch.int32),
        accept_index=empty,
    )
    assert not calls


@pytest.mark.parametrize(
    "field", ["predict_rows", "logits_rank", "batch_rows", "devices"]
)
def test_inconsistent_layout_is_rejected_before_upstream(guard, field):
    batch, logits, predict, index = inputs()
    if field == "predict_rows":
        predict = predict[:-1]
    elif field == "logits_rank":
        logits.next_token_logits = logits.next_token_logits.flatten()
    elif field == "batch_rows":
        batch.seq_lens = batch.seq_lens[:-1]
    else:
        index = torch.empty((4, 4), dtype=torch.int32, device="meta")
    original, calls = spy()
    with pytest.raises(ValueError):
        partial(guard._compute_spec_logprobs_hook, original)(
            batch, logits, predict, accept_index=index
        )
    assert not calls


def test_chain_layout_is_stable_and_other_platforms_are_exact_passthrough(
    guard, monkeypatch
):
    original, calls = spy()
    batch, logits, predict, _ = inputs()
    guard._compute_spec_logprobs_hook(original, batch, logits, predict, chain_stride=4)
    expected = (
        torch.nn.functional.log_softmax(logits.next_token_logits.double(), dim=-1)
        .gather(1, predict.long()[:, None])
        .view(4, 4)
    )
    torch.testing.assert_close(
        logits.next_token_logprobs.double(), expected, rtol=1e-5, atol=2e-5
    )
    assert not calls
    monkeypatch.delattr(torch, "__hcu_version__")
    opaque = object()
    guard._compute_spec_logprobs_hook(
        original, opaque, opaque, opaque, accept_index=opaque
    )
    assert calls[-1] == (opaque, opaque, opaque, opaque, None)


def test_marker_is_emitted_only_after_real_normalization_once(guard, caplog):
    batch, logits, predict, index = inputs()
    original, _ = spy()
    wrapped = partial(guard._compute_spec_logprobs_hook, original)
    with caplog.at_level("INFO"):
        wrapped(batch, logits, predict, accept_index=index.clamp(min=0))
        assert not any("padding normalized" in r.message for r in caplog.records)
        wrapped(batch, logits, predict, accept_index=index)
        wrapped(batch, logits, predict, accept_index=index)
    messages = [
        r.message
        for r in caplog.records
        if "HCU MTP logprob padding normalized" in r.message
    ]
    assert len(messages) == 1
    assert "padding_count=2" in messages[0] and "last_row=15" in messages[0]


@pytest.mark.parametrize("original_logprobs", [False, True])
def test_temperature_original_flag_topk_and_optional_token_ids(
    guard, original_logprobs
):
    batch, logits, predict, index = inputs()
    batch.sampling_info.is_all_greedy = False
    batch.sampling_info.temperatures = torch.tensor([[0.5], [1.5], [2.0], [0.8]])
    batch.top_logprobs_nums = [2, 0, 3, 1]
    batch.token_ids_logprobs = [[0, 2], None, [], [6]]
    guard._test_processor.return_original = original_logprobs
    original, calls = spy()
    guard._compute_spec_logprobs_hook(
        original, batch, logits, predict, accept_index=index
    )
    selected = logits.next_token_logits[index.long().flatten()]
    if not original_logprobs:
        selected = selected / batch.sampling_info.temperatures.repeat_interleave(
            4, dim=0
        )
    expected = torch.nn.functional.log_softmax(selected.double(), dim=-1)
    chosen = expected.gather(1, predict[index.long().flatten()].long()[:, None]).view(
        4, 4
    )
    torch.testing.assert_close(
        logits.next_token_logprobs.double(), chosen, rtol=1e-5, atol=2e-5
    )
    assert guard._test_processor.top_requests == [
        [n for n in batch.top_logprobs_nums for _ in range(4)]
    ]
    assert guard._test_processor.token_requests == [
        [ids for ids in batch.token_ids_logprobs for _ in range(4)]
    ]
    for row, values in enumerate(logits.next_token_top_logprobs_val):
        n = batch.top_logprobs_nums[row // 4]
        torch.testing.assert_close(
            values.double(), expected[row].topk(n).values, rtol=1e-5, atol=2e-5
        )
    for row, values in enumerate(logits.next_token_token_ids_logprobs_val):
        ids = batch.token_ids_logprobs[row // 4]
        if ids is not None:
            torch.testing.assert_close(
                values.double(), expected[row, ids], rtol=1e-5, atol=2e-5
            )
    assert not calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_upstream_output_dtype_and_device_preserved(guard, dtype):
    batch, logits, predict, index = inputs()
    logits.next_token_logits = logits.next_token_logits.to(dtype)
    original, _ = spy()
    guard._compute_spec_logprobs_hook(
        original, batch, logits, predict, accept_index=index
    )
    assert logits.next_token_logprobs.dtype == dtype
    assert logits.next_token_logprobs.device == logits.next_token_logits.device


def test_half_clamp_and_large_row_offset_preserve_stable_subtractions(guard):
    original, _ = spy()
    batch = SimpleNamespace(
        seq_lens=[1],
        sampling_info=SimpleNamespace(is_all_greedy=True),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits = SimpleNamespace(
        next_token_logits=torch.tensor([[65504.0, -65504.0]], dtype=torch.float16)
    )
    guard._compute_spec_logprobs_hook(
        original, batch, logits, torch.tensor([1]), chain_stride=1
    )
    assert logits.next_token_logprobs.item() == torch.finfo(torch.float16).min
    logits = SimpleNamespace(
        next_token_logits=torch.tensor([[1e8, 1e8]], dtype=torch.float32)
    )
    guard._compute_spec_logprobs_hook(
        original, batch, logits, torch.tensor([0]), chain_stride=1
    )
    torch.testing.assert_close(
        logits.next_token_logprobs, torch.tensor([[-0.69314718]])
    )


@pytest.mark.parametrize("layout", ["neither", "both", "wrong_chain_rows"])
def test_layout_exclusivity_and_chain_row_contract(guard, layout):
    batch, logits, predict, index = inputs()
    original, calls = spy()
    kwargs = (
        {}
        if layout == "neither"
        else {"accept_index": index, "chain_stride": 4}
        if layout == "both"
        else {"chain_stride": 3}
    )
    with pytest.raises(AssertionError):
        guard._compute_spec_logprobs_hook(original, batch, logits, predict, **kwargs)
    assert not calls


@pytest.fixture
def registry_module(monkeypatch):
    override = os.getenv("SGLANG_HCU_UPSTREAM_SOURCE")
    if override:
        root = Path(override)
        candidates = (root / "python/sglang", root / "sglang", root)
    else:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            pytest.skip(
                "installed SGLang or explicit SGLANG_HCU_UPSTREAM_SOURCE required for real hook-registry integration"
            )
        candidates = (Path(spec.origin).parent,)
    relative = "srt/plugins/hook_registry.py"
    path = next(
        (root / relative for root in candidates if (root / relative).is_file()), None
    )
    if path is None:
        pytest.fail(f"could not find actual SGLang hook registry under {candidates}")
    spec = importlib.util.spec_from_file_location(
        "isolated_actual_sg_hook_registry", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.plugins",
        "sglang.srt.layers",
        "sglang.srt.speculative",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "sglang.srt.plugins.hook_registry", module)
    return module


def test_real_registry_propagates_to_already_loaded_common_alias(
    guard, registry_module, monkeypatch
):
    original, _ = spy()
    modules = []
    for name in (TARGET, COMMON):
        module = ModuleType(name)
        module.__dict__.update(guard._test_processor.__dict__)
        module.__name__ = name
        module.compute_spec_logprobs = original
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    assert guard.register_hcu_mtp_logprob_guard()
    assert len(registry_module.HookRegistry._hooks[guard._TARGET]) == 1
    registry_module.HookRegistry.apply_hooks()
    wrapped = modules[0].compute_spec_logprobs
    assert modules[1].compute_spec_logprobs is wrapped
    assert wrapped.__wrapped__ is original
    assert guard._TARGET in registry_module.HookRegistry._patched
    batch, logits, predict, index = inputs()
    assert (
        modules[1].compute_spec_logprobs(batch, logits, predict, accept_index=index)
        is None
    )
    assert logits.next_token_logprobs.shape == (4, 4)


def test_real_registry_base_first_then_common_natural_import(
    guard, registry_module, monkeypatch
):
    monkeypatch.delitem(sys.modules, COMMON, raising=False)
    base = ModuleType(TARGET)
    base.__dict__.update(guard._test_processor.__dict__)
    base.compute_spec_logprobs = spy()[0]
    monkeypatch.setitem(sys.modules, TARGET, base)
    guard.register_hcu_mtp_logprob_guard()
    registry_module.HookRegistry.apply_hooks()

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.compute_spec_logprobs = importlib.import_module(
                TARGET
            ).compute_spec_logprobs

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            return (
                importlib.util.spec_from_loader(fullname, Loader())
                if fullname == COMMON
                else None
            )

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        common = importlib.import_module(COMMON)
        assert common.compute_spec_logprobs is base.compute_spec_logprobs
        batch, logits, predict, index = inputs()
        assert (
            common.compute_spec_logprobs(batch, logits, predict, accept_index=index)
            is None
        )
        assert logits.next_token_logprobs.shape == (4, 4)

    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(COMMON, None)


def test_registration_is_lazy_idempotent_and_hcu_only(guard, monkeypatch):
    registered = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookRegistry = SimpleNamespace(
        register=lambda *args: registered.append(args)
    )
    registry.HookType = SimpleNamespace(AROUND="around")
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name in (TARGET, COMMON) or name == "torch":
            raise AssertionError("registration imported the target or Torch eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    for _ in range(3):
        assert guard.register_hcu_mtp_logprob_guard()
    assert registered == [(guard._TARGET, guard._compute_spec_logprobs_hook, "around")]
    monkeypatch.delattr(torch, "__hcu_version__")
    assert not guard.register_hcu_mtp_logprob_guard()
    assert len(registered) == 1


def test_simultaneous_registration_adds_exactly_one_hook(guard, monkeypatch):
    registered = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookRegistry = SimpleNamespace(
        register=lambda *args: registered.append(args)
    )
    registry.HookType = SimpleNamespace(AROUND="around")
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    barrier = Barrier(3)
    errors = []

    def register():
        try:
            barrier.wait(3)
            guard.register_hcu_mtp_logprob_guard()
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=register) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(3)
    for thread in threads:
        thread.join(3)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert registered == [(guard._TARGET, guard._compute_spec_logprobs_hook, "around")]
