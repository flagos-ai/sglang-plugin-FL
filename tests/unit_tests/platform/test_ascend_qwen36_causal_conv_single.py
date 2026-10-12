# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import (
    qwen36_causal_conv as conv_single,
)

_SingleSequenceMetadata = conv_single._SingleSequenceMetadata
_get_single_sequence_metadata = conv_single._get_prefill_metadata
_make_optimized_causal_conv1d_fn = conv_single._make_optimized_causal_conv1d_fn
_native_api_supported = conv_single._native_api_supported
_BACKEND_MODULE = conv_single._BACKEND_MODULE
_run_single_sequence = conv_single._run_single_sequence
patch_qwen36_causal_conv_prefill = conv_single.patch_qwen36_causal_conv_prefill


def _target_inputs(dim: int = 5120, valid_tokens: int = 5, tail: int = 0):
    width = 4
    x = torch.empty(dim, valid_tokens + tail, dtype=torch.bfloat16)
    weight = torch.empty(dim, width, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, valid_tokens], dtype=torch.int32)
    cache_indices = torch.tensor([1], dtype=torch.int64)
    has_initial_state = torch.tensor([False], dtype=torch.bool)
    conv_states = torch.empty(3, dim, width - 1, dtype=torch.bfloat16)
    return (
        x,
        weight,
        None,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        "silu",
        -1,
        [valid_tokens],
    )


def test_guards_accept_both_qwen36_shapes_and_tail_padding():
    for dim in (4096, 5120):
        metadata = _get_single_sequence_metadata(*_target_inputs(dim, tail=2))
        assert metadata == _SingleSequenceMetadata(5, 1, False)


def test_guards_accept_initial_state_continuation():
    args = list(_target_inputs())
    args[5] = torch.tensor([True], dtype=torch.bool)
    metadata = _get_single_sequence_metadata(*args)
    assert metadata == _SingleSequenceMetadata(5, 1, True)


def test_pad_slot_and_invalid_cache_force_fallback():
    args = list(_target_inputs())
    args[4] = torch.tensor([-1], dtype=torch.int64)
    assert _get_single_sequence_metadata(*args) is None

    args = list(_target_inputs())
    args[4] = torch.tensor([3], dtype=torch.int64)
    assert _get_single_sequence_metadata(*args) is None


def test_pad_slot_wrapper_calls_upstream_without_running_native():
    calls = []
    sentinel = object()

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def native(*args, **kwargs):
        raise AssertionError("native fast path must not run for a pad slot")

    args = _target_inputs()
    optimized = _make_optimized_causal_conv1d_fn(upstream, native)
    result = optimized(
        args[0],
        args[1],
        args[2],
        query_start_loc=args[3],
        cache_indices=torch.tensor([-1], dtype=torch.int64),
        has_initial_state=args[5],
        conv_states=args[6],
        activation=args[7],
        pad_slot_id=args[8],
        seq_lens_cpu=args[9],
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][1]["pad_slot_id"] == -1


def test_non_target_shapes_and_metadata_force_fallback():
    args = list(_target_inputs())
    args[1] = torch.empty(5120, 3, dtype=torch.bfloat16)
    assert _get_single_sequence_metadata(*args) is None

    args = list(_target_inputs())
    args[3] = torch.tensor([0, 2, 5], dtype=torch.int32)
    args[4] = torch.tensor([0, 1], dtype=torch.int64)
    args[5] = torch.tensor([False, False], dtype=torch.bool)
    assert _get_single_sequence_metadata(*args) is None

    args = list(_target_inputs())
    args[-1] = [4]
    assert _get_single_sequence_metadata(*args) is None


def test_seq_lens_cpu_rejects_lossy_or_boolean_coercions():
    for invalid_length in (5.9, "5", True):
        args = list(_target_inputs())
        args[-1] = [invalid_length]
        assert _get_single_sequence_metadata(*args) is None


def _check_cpu_equivalence(native_fn, reference_fn, use_initial_state, tail):
    if not _native_api_supported(native_fn):
        pytest.skip("SDK lacks direct final-state writeback; wrapper uses fallback")
    torch.manual_seed(23)
    dim, valid_tokens, width = 8, 7, 4
    x = torch.randn(dim, valid_tokens + tail, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)
    states = torch.randn(3, dim, width - 1, dtype=torch.bfloat16)
    expected_states = states.clone()
    actual_states = states.clone()
    state = expected_states[1].unsqueeze(0)
    expected, _ = reference_fn(
        x[..., :valid_tokens],
        weight,
        bias,
        initial_states=state if use_initial_state else None,
        return_final_states=True,
        final_states_out=state,
        activation="silu",
    )
    if tail:
        expected = torch.nn.functional.pad(expected, (0, tail))

    actual = _run_single_sequence(
        native_fn,
        x,
        weight,
        bias,
        actual_states,
        "silu",
        _SingleSequenceMetadata(valid_tokens, 1, use_initial_state),
    )
    assert torch.equal(actual, expected)
    assert torch.equal(actual_states, expected_states)
    if tail:
        assert torch.count_nonzero(actual[..., valid_tokens:]) == 0


def test_output_and_state_match_without_initial_state(
    native_causal_conv, cpu_causal_conv
):
    _check_cpu_equivalence(native_causal_conv, cpu_causal_conv, False, 0)


def test_output_and_state_match_with_initial_state_and_tail_padding(
    native_causal_conv, cpu_causal_conv
):
    _check_cpu_equivalence(native_causal_conv, cpu_causal_conv, True, 3)


def test_target_b1_wrapper_calls_native_once():
    native_calls = []

    def upstream(*args, **kwargs):
        raise AssertionError("upstream wrapper must not run for the target B=1 path")

    def native(
        *args,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        native_calls.append(
            (
                args,
                {
                    "initial_states": initial_states,
                    "return_final_states": return_final_states,
                    "final_states_out": final_states_out,
                    "activation": activation,
                },
            )
        )
        return args[0], final_states_out

    args = _target_inputs()
    optimized = _make_optimized_causal_conv1d_fn(upstream, native)
    result = optimized(
        args[0],
        args[1],
        args[2],
        query_start_loc=args[3],
        cache_indices=args[4],
        has_initial_state=args[5],
        conv_states=args[6],
        activation=args[7],
        pad_slot_id=args[8],
        seq_lens_cpu=args[9],
    )

    assert len(native_calls) == 1
    native_args, native_kwargs = native_calls[0]
    assert result is native_args[0]
    assert native_args[0].shape == args[0].shape
    assert native_kwargs["initial_states"] is None
    assert native_kwargs["return_final_states"] is True
    assert native_kwargs["final_states_out"].shape == (1, 5120, 3)
    assert native_kwargs["activation"] == "silu"


def test_native_api_missing_final_states_out_falls_back_only_for_single(mock_npu_mamba):
    sentinel = object()
    calls = []

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def native_without_final_states_out(
        x,
        weight,
        bias,
        *,
        initial_states=None,
        return_final_states=False,
        activation=None,
    ):
        return x, None

    causal_conv = SimpleNamespace(
        causal_conv1d_fn_npu=upstream,
        causal_conv1d_fn_native=native_without_final_states_out,
    )
    backend = SimpleNamespace(
        causal_conv1d_fn_npu=upstream,
        causal_conv1d_fn=upstream,
    )
    with (
        mock.patch.object(mock_npu_mamba, "causal_conv1d", causal_conv, create=True),
        mock.patch.dict(sys.modules, {_BACKEND_MODULE: backend}),
    ):
        assert not _native_api_supported(native_without_final_states_out)
        assert patch_qwen36_causal_conv_prefill() is True
        optimized = causal_conv.causal_conv1d_fn_npu
        args = _target_inputs()
        assert (
            optimized(
                *args[:3],
                query_start_loc=args[3],
                cache_indices=args[4],
                has_initial_state=args[5],
                conv_states=args[6],
                seq_lens_cpu=args[9],
            )
            is sentinel
        )
        assert len(calls) == 1
        assert backend.causal_conv1d_fn_npu is optimized
        assert backend.causal_conv1d_fn is optimized
        # The same native API still supports multi-sequence batching.
        from sglang_fl.dispatch.backends.vendor.ascend.patches import qwen36_causal_conv

        with mock.patch.object(
            qwen36_causal_conv, "_run_equal_length_batched", return_value="batched"
        ):
            assert (
                optimized(
                    args[0],
                    args[1],
                    query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
                    cache_indices=torch.tensor([0, 1]),
                    has_initial_state=torch.tensor([False, False]),
                    conv_states=args[6],
                    seq_lens_cpu=[2, 2],
                )
                == "batched"
            )
        assert len(calls) == 1


def test_patch_rebinds_loaded_backend_aliases_and_is_idempotent(mock_npu_mamba):
    def upstream(*args, **kwargs):
        return None

    def native(
        x,
        weight,
        bias,
        *,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        return x, final_states_out

    causal_conv = SimpleNamespace(
        causal_conv1d_fn_npu=upstream,
        causal_conv1d_fn_native=native,
    )
    backend = SimpleNamespace(
        causal_conv1d_fn_npu=upstream,
        causal_conv1d_fn=upstream,
    )
    with (
        mock.patch.object(mock_npu_mamba, "causal_conv1d", causal_conv, create=True),
        mock.patch.dict(sys.modules, {_BACKEND_MODULE: backend}),
    ):
        assert patch_qwen36_causal_conv_prefill() is True
        optimized = causal_conv.causal_conv1d_fn_npu
        assert backend.causal_conv1d_fn_npu is optimized
        assert backend.causal_conv1d_fn is optimized
        assert patch_qwen36_causal_conv_prefill() is True
        assert causal_conv.causal_conv1d_fn_npu is optimized
        assert backend.causal_conv1d_fn_npu is optimized
        assert backend.causal_conv1d_fn is optimized


def test_non_target_wrapper_preserves_all_arguments():
    calls = []
    sentinel = object()

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    def native(*args, **kwargs):
        raise AssertionError("native fast path must not run")

    optimized = _make_optimized_causal_conv1d_fn(upstream, native)
    result = optimized(
        torch.empty(8, 6),
        torch.empty(8, 4),
        query_start_loc=torch.tensor([0, 3, 6]),
        cache_indices=torch.tensor([0, 1]),
        has_initial_state=torch.tensor([False, False]),
        conv_states=torch.empty(2, 8, 3),
        pad_slot_id=-7,
        seq_lens_cpu=[3, 3],
        future_argument="preserved",
    )

    assert result is sentinel
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["pad_slot_id"] == -7
    assert kwargs["seq_lens_cpu"] == [3, 3]
    assert kwargs["future_argument"] == "preserved"
