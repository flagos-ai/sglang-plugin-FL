# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real HCU/FlagGems speculative logprobs against a CPU negative-index oracle.

No unsafe negative advanced-index operation is launched on the accelerator.
The production guard first converts the documented -1 padding to N-1. These
are component regressions; model-level batch acceptance remains separate.
"""

from contextlib import contextmanager
import importlib
import inspect
import math
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.functional, pytest.mark.gpu]
VOCAB = 248320
PREDICT = [
    15,
    248046,
    198,
    248044,
    248046,
    198,
    248044,
    248045,
    12305,
    198,
    727,
    51184,
    17,
    16,
    0,
    0,
]


@pytest.fixture
def hcu_logprobs():
    if not hasattr(torch, "__hcu_version__") or not torch.cuda.is_available():
        pytest.skip("real Hygon accelerator required")
    import flag_gems
    from sglang_fl.dispatch.backends.vendor.hygon.patches import mtp_logprobs
    from sglang.srt.plugins.hook_registry import HookRegistry

    mtp_logprobs.register_hcu_mtp_logprob_guard()
    HookRegistry.apply_hooks()
    base = importlib.import_module("sglang.srt.layers.logprob_processor")
    common = importlib.import_module("sglang.srt.speculative.eagle_worker_common")
    assert mtp_logprobs._TARGET in HookRegistry._patched
    assert common.compute_spec_logprobs is base.compute_spec_logprobs
    original = inspect.unwrap(base.compute_spec_logprobs)
    assert original.__module__ == "sglang.srt.layers.logprob_processor"
    return mtp_logprobs, flag_gems, original, common.compute_spec_logprobs


@contextmanager
def actual_gems_calls():
    # Prove actual Python operator dispatch instead of just checking the device.
    index = importlib.import_module("flag_gems.ops.index").index
    gather = importlib.import_module("flag_gems.ops.gather").gather
    row = importlib.import_module("sglang.srt.layers.logsumexp").row_logsumexp
    softmax = importlib.import_module("flag_gems.ops.log_softmax").log_softmax
    counts = {"index": 0, "gather": 0, "SG_row_logsumexp": 0, "Gems_log_softmax": 0}
    codes = {
        index.__code__: "index",
        gather.__code__: "gather",
        row.__code__: "SG_row_logsumexp",
        softmax.__code__: "Gems_log_softmax",
    }
    previous = sys.getprofile()
    assert previous is None, "existing profiler makes vendor attribution ambiguous"

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_hcu_padding_equivalence_for_real_index_kernels(hcu_logprobs, rows, dtype):
    patch, gems, _, _ = hcu_logprobs
    cpu_source = torch.arange(rows, dtype=dtype) * 7 + 3
    cpu_index = torch.tensor([[-1]], dtype=torch.int32)
    expected = cpu_source[cpu_index.long()]
    source = cpu_source.to("cuda")
    index = cpu_index.to("cuda")
    with gems.use_gems(), actual_gems_calls() as calls:
        positive = patch.normalize_hcu_spec_logprob_accept_index(index, rows)
        actual = source[positive.long()]
        torch.cuda.synchronize()
    assert calls["index"] > 0
    assert positive.item() == rows - 1
    assert index.item() == -1
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def fixture_values(accepted):
    logits = torch.full((16, VOCAB), -12.0, dtype=torch.float32)
    for row in range(16):
        columns = [(31 * row + offset) % VOCAB for offset in (0, 17, 37, 71, 113)]
        logits[row, columns] = torch.tensor([4.3, 3.2, 2.1, 1.0, 0.2]) + row * 0.03
    predict = torch.tensor(PREDICT, dtype=torch.int32)
    index = torch.full((4, 4), -1, dtype=torch.int32)
    for row, length in enumerate(accepted):
        index[row, :length] = torch.arange(row * 4, row * 4 + length, dtype=torch.int32)
    return logits, predict, index


def oracle(logits, predict, index):
    # Run all negative-index operations only on CPU, outside use_gems.
    flat = index.long().flatten()
    selected_logits = logits[flat].double()
    selected_tokens = predict[flat].long()
    logprobs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
    chosen = logprobs.gather(1, selected_tokens[:, None]).view(4, 4)
    top_values, top_indices = logprobs.topk(5, dim=-1)
    normalizer = torch.logsumexp(selected_logits, dim=-1)
    # Independent analytic sum of the actual stored FP32 values, promoted to
    # double. Do not reconstruct ideal decimal peak values or change the input.
    analytic = []
    for row in selected_logits:
        values, counts = torch.unique(row, return_counts=True)
        maximum = values.max().item()
        analytic.append(
            maximum
            + math.log(
                math.fsum(
                    int(n) * math.exp(float(v) - maximum)
                    for v, n in zip(values.tolist(), counts.tolist())
                )
            )
        )
    torch.testing.assert_close(
        normalizer, torch.tensor(analytic, dtype=torch.float64), rtol=0, atol=1e-10
    )
    return chosen, top_values, top_indices, normalizer


def output(logits):
    return SimpleNamespace(next_token_logits=logits)


def check_output(result, expected):
    chosen, top_values, top_indices, _ = expected
    torch.testing.assert_close(
        result.next_token_logprobs.cpu().double(), chosen, rtol=1e-5, atol=2e-5
    )
    actual_values = torch.stack(result.next_token_top_logprobs_val).cpu()
    actual_indices = torch.stack(result.next_token_top_logprobs_idx).cpu()
    torch.testing.assert_close(actual_values.double(), top_values, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_indices, top_indices, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [(4, 4, 4, 2), (1, 2, 4, 3), (4, 4, 4, 4)])
def test_hcu_real_spec_logprobs_preserve_tokens_top5_and_normalizers(
    hcu_logprobs, accepted, monkeypatch
):
    patch, gems, original, registered = hcu_logprobs
    cpu_logits, cpu_predict, cpu_index = fixture_values(accepted)
    expected = oracle(cpu_logits, cpu_predict, cpu_index)
    logits, predict, index = (
        tensor.to("cuda") for tensor in (cpu_logits, cpu_predict, cpu_index)
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([27, 25, 40, 28], device="cuda"),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        top_logprobs_nums=[5] * 4,
        token_ids_logprobs=None,
    )
    original_calls, seen_inputs, normalizers = [], [], []
    processor = importlib.import_module("sglang.srt.layers.logprob_processor")
    actual_row_normalizer = processor.compute_row_log_normalizer

    def observed_normalizer(selected):
        maximum, shifted = actual_row_normalizer(selected)
        seen_inputs.append(selected.detach().cpu().clone())
        normalizers.append(
            maximum.detach().cpu().double() + shifted.detach().cpu().double()
        )
        return maximum, shifted

    monkeypatch.setattr(processor, "compute_row_log_normalizer", observed_normalizer)

    def observed_original(*args, **kwargs):
        original_calls.append(True)
        return original(*args, **kwargs)

    component, through_registry = output(logits), output(logits)
    with gems.use_gems(), actual_gems_calls() as calls:
        patch._compute_spec_logprobs_hook(
            observed_original, batch, component, predict, accept_index=index
        )
        registered(batch, through_registry, predict, accept_index=index)
        torch.cuda.synchronize()
    assert calls["index"] >= 4 and calls["gather"] >= 2
    assert calls["SG_row_logsumexp"] >= 2 and calls["Gems_log_softmax"] == 0
    assert not original_calls
    expected_rows = cpu_logits[cpu_index.long().flatten()]
    assert len(seen_inputs) == 2 and all(
        torch.equal(row, expected_rows) for row in seen_inputs
    )
    assert torch.equal(index.cpu(), cpu_index)
    assert torch.equal(predict.cpu(), cpu_predict)
    check_output(component, expected)
    check_output(through_registry, expected)
    for normalizer in normalizers:
        torch.testing.assert_close(normalizer, expected[3], rtol=1e-5, atol=2e-5)


def test_hcu_chain_stride_preserves_probability_semantics(hcu_logprobs):
    _, gems, _, registered = hcu_logprobs
    cpu_logits, cpu_predict, cpu_index = fixture_values((4, 4, 4, 4))
    expected = oracle(cpu_logits, cpu_predict, cpu_index)
    logits, predict = cpu_logits.to("cuda"), cpu_predict.to("cuda")
    batch = SimpleNamespace(
        seq_lens=torch.arange(4, device="cuda"),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        top_logprobs_nums=[5] * 4,
        token_ids_logprobs=None,
    )
    result = output(logits)
    with gems.use_gems(), actual_gems_calls() as calls:
        registered(batch, result, predict, chain_stride=4)
        torch.cuda.synchronize()
    assert calls["gather"] >= 1
    assert calls["SG_row_logsumexp"] >= 1 and calls["Gems_log_softmax"] == 0
    check_output(result, expected)
