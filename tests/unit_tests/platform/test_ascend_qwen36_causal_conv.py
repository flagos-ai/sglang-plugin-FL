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

import inspect

import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen36_causal_conv import (
    _BatchMetadata,
    _SingleSequenceMetadata,
    _get_prefill_metadata,
    _make_optimized_causal_conv1d_fn,
    _run_equal_length_batched,
)


def _reference_per_sequence(
    native_fn,
    x,
    weight,
    query_start_loc,
    cache_indices,
    has_initial_state,
    conv_states,
):
    outputs = []
    for i in range(cache_indices.numel()):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        state_index = int(cache_indices[i])
        initial_state = (
            conv_states[state_index : state_index + 1]
            if bool(has_initial_state[i])
            else None
        )
        out, final_state = native_fn(
            x[:, start:end],
            weight,
            None,
            initial_states=initial_state,
            return_final_states=True,
            activation="silu",
        )
        conv_states[state_index : state_index + 1].copy_(final_state)
        outputs.append(out)
    return torch.cat(outputs, dim=-1)


def _run_cpu_equivalence(native_fn, reference_fn, use_initial_state, trailing_tokens):
    torch.manual_seed(17)
    batch, dim, seq_len, width = 3, 8, 5, 4
    valid_tokens = batch * seq_len
    x = torch.randn(dim, valid_tokens + trailing_tokens, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    query_start_loc = torch.arange(0, valid_tokens + 1, seq_len, dtype=torch.int32)
    cache_indices = torch.tensor([2, 0, 3], dtype=torch.int64)
    has_initial_state = torch.full((batch,), use_initial_state, dtype=torch.bool)
    base_states = torch.randn(5, dim, width - 1, dtype=torch.bfloat16)

    expected_states = base_states.clone()
    actual_states = base_states.clone()
    expected = _reference_per_sequence(
        reference_fn,
        x[:, :valid_tokens],
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        expected_states,
    )
    if trailing_tokens:
        expected = torch.nn.functional.pad(expected, (0, trailing_tokens))

    metadata = _BatchMetadata(batch, seq_len, valid_tokens, use_initial_state)
    actual = _run_equal_length_batched(
        native_fn,
        x,
        weight,
        None,
        cache_indices,
        actual_states,
        "silu",
        metadata,
        native_metadata="seqlens" in inspect.signature(native_fn).parameters,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual_states, expected_states)
    if trailing_tokens:
        assert torch.count_nonzero(actual[:, valid_tokens:]) == 0


def test_batched_path_matches_without_initial_state(
    native_causal_conv, cpu_causal_conv
):
    _run_cpu_equivalence(native_causal_conv, cpu_causal_conv, False, 0)


def test_batched_path_matches_with_initial_state_and_tail_padding(
    native_causal_conv, cpu_causal_conv
):
    _run_cpu_equivalence(native_causal_conv, cpu_causal_conv, True, 2)


def _target_inputs(dim: int = 5120):
    batch, seq_len, width = 2, 4, 4
    x = torch.empty(dim, batch * seq_len, dtype=torch.bfloat16)
    weight = torch.empty(dim, width, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, seq_len, batch * seq_len], dtype=torch.int32)
    cache_indices = torch.tensor([0, 2], dtype=torch.int64)
    has_initial_state = torch.zeros(batch, dtype=torch.bool)
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
    )


def test_shape_guards_accept_both_qwen36_models() -> None:
    assert _get_prefill_metadata(*_target_inputs(5120)) is not None
    assert _get_prefill_metadata(*_target_inputs(4096)) is not None


def test_cpu_seq_lens_mirror_avoids_query_start_loc_value_reads() -> None:
    args = list(_target_inputs())

    class UnreadableQueryStartLoc:
        ndim = 1
        dtype = torch.int32
        device = args[0].device

        @staticmethod
        def numel() -> int:
            return 3

        def __getitem__(self, key):
            raise AssertionError("CPU mirror path must not read device values")

    args[3] = UnreadableQueryStartLoc()
    metadata = _get_prefill_metadata(*args, seq_lens_cpu=[4, 4])
    assert metadata == _BatchMetadata(2, 4, 8, False)


def test_invalid_cpu_seq_lens_mirror_forces_fallback() -> None:
    args = _target_inputs()
    assert _get_prefill_metadata(*args, seq_lens_cpu=[3, 5]) is None
    assert _get_prefill_metadata(*args, seq_lens_cpu=[4]) is None
    assert _get_prefill_metadata(*args, seq_lens_cpu=[4, True]) is None
    assert _get_prefill_metadata(*args, seq_lens_cpu=None) is not None


def test_shared_guards_reject_other_shape_and_accept_single_sequence() -> None:
    args = list(_target_inputs())
    args[1] = torch.empty(5120, 3, dtype=torch.bfloat16)
    assert _get_prefill_metadata(*args) is None

    args = list(_target_inputs())
    args[3] = torch.tensor([0, 8], dtype=torch.int32)
    args[4] = torch.tensor([0], dtype=torch.int64)
    args[5] = torch.tensor([False])
    assert _get_prefill_metadata(*args) == _SingleSequenceMetadata(8, 0, False)


def test_branch_guards_reject_unequal_or_mixed_state_batches() -> None:
    args = list(_target_inputs())
    args[3] = torch.tensor([0, 3, 8], dtype=torch.int32)
    assert _get_prefill_metadata(*args) is None

    args = list(_target_inputs())
    args[5] = torch.tensor([False, True])
    assert _get_prefill_metadata(*args) is None


def test_pad_slot_guard_forces_fallback() -> None:
    args = list(_target_inputs())
    args[4] = torch.tensor([0, -1], dtype=torch.int64)
    assert _get_prefill_metadata(*args) is None


def test_non_target_wrapper_forwards_signature_and_kwargs() -> None:
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
        future_argument="preserved",
        seq_lens_cpu=[3, 3],
    )

    assert result is sentinel
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["pad_slot_id"] == -7
    assert kwargs["future_argument"] == "preserved"
    assert kwargs["seq_lens_cpu"] == [3, 3]
