# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tools.operator_profiling.workload import (
    build_exact_token_ids,
    resolve_lengths,
    resolve_runtime_reserved_tokens,
    validate_generation_result,
)


class _Tokenizer:
    bos_token_id = 1

    def encode(self, _text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10, 11, 12]


@pytest.mark.parametrize("length", [1, 2, 7, 1024])
def test_exact_token_builder(length):
    tokens = build_exact_token_ids(_Tokenizer(), length)
    assert len(tokens) == length
    assert tokens[0] == 1


def test_length_resolution_preserves_output_and_truncates_input():
    result = resolve_lengths(262144, 1024, 262144)
    assert result.actual_input_tokens == 261120
    assert result.actual_output_tokens == 1024
    assert result.input_truncated is True
    assert result.truncated_token_count == 1024


def test_runtime_request_limit_contributes_a_conservative_margin():
    reserved = resolve_runtime_reserved_tokens(
        262144,
        framework_reserved_tokens=0,
        max_req_input_len=262138,
    )
    assert reserved == 6

    result = resolve_lengths(262144, 1024, 262144, reserved_tokens=reserved)
    assert result.actual_input_tokens == 261114
    assert result.actual_output_tokens == 1024


def test_runtime_reservation_keeps_larger_model_specific_margin():
    assert (
        resolve_runtime_reserved_tokens(
            100,
            framework_reserved_tokens=8,
            max_req_input_len=96,
        )
        == 8
    )


@pytest.mark.parametrize("max_req_input_len", [0, 101])
def test_runtime_request_limit_must_fit_model_context(max_req_input_len):
    with pytest.raises(ValueError, match="max_req_input_len"):
        resolve_runtime_reserved_tokens(100, max_req_input_len=max_req_input_len)


def test_length_resolution_accounts_for_reserved_tokens():
    result = resolve_lengths(100, 20, 110, reserved_tokens=4)
    assert result.actual_input_tokens == 86
    assert result.truncated_token_count == 14


def test_length_resolution_error_policy_is_explicit():
    with pytest.raises(ValueError, match="requested 100 input"):
        resolve_lengths(100, 20, 110, overflow_policy="error")


def test_generation_validation_checks_batch_and_exact_output_length():
    rows = [{"meta_info": {"completion_tokens": 8}} for _ in range(4)]
    assert validate_generation_result(rows, concurrency=4, output_tokens=8) == [8] * 4
    with pytest.raises(RuntimeError, match="expected exactly 7"):
        validate_generation_result(rows, concurrency=4, output_tokens=7)
