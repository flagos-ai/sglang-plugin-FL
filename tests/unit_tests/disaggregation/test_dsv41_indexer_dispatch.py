# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only dispatch, policy and failure contracts for V4.1 indexers."""

import os
import sys
import types

import pytest

from sglang_fl.dispatch import SelectionPolicy, policy_context
from sglang_fl.dispatch.backends.flagos.register_ops import register_builtins as flagos
from sglang_fl.dispatch.backends.reference.register_ops import (
    register_builtins as reference,
)
from sglang_fl.dispatch.manager import OpManager
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dsv41_runtime import indexer


@pytest.fixture
def route(monkeypatch):
    calls = []
    native_result, base_result, long_result = object(), object(), object()
    state = types.SimpleNamespace(mode="decode", outcome=long_result)

    def native(*args, **kwargs):
        calls.append(("native", args, kwargs))
        return native_result

    def base(*args, **kwargs):
        calls.append(("base", args, kwargs))
        return base_result

    def optimized(*args, **kwargs):
        calls.append(("long", args, kwargs))
        if isinstance(state.outcome, Exception):
            raise state.outcome
        return state.outcome

    funcs = {
        "fp4_index_logits_req_to_token": base,
        "fp4_index_logits_candidate_blocks": base,
        "dsv41_long_context_producer": optimized,
        "dsv41_long_context_compact": optimized,
    }
    gems = types.ModuleType("flaggems_sglang")
    gems.get_op = funcs.get
    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_disagg = lambda: types.SimpleNamespace(disaggregation_mode=state.mode)
    for name in ("sglang", "sglang.srt"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, gems.__name__, gems)

    registry = OpRegistry()
    flagos(registry)
    reference(registry)
    manager = OpManager(registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    monkeypatch.setattr("sglang_fl.dispatch.get_default_manager", lambda: manager)
    return state, calls, native, native_result, base_result, long_result, funcs


@pytest.mark.parametrize("method", [indexer.producer, indexer.compact])
def test_decode_hits_long_path_and_forwards_arguments(route, method):
    _, calls, native, _, _, long_result, _ = route
    token = object()
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert method(native, token, x=7) is long_result
    assert calls == [("long", (token,), {"x": 7})]


@pytest.mark.parametrize("method", [indexer.producer, indexer.compact])
def test_prefill_uses_standard_indexer(route, method):
    state, calls, native, _, base_result, _, _ = route
    state.mode = "prefill"
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert method(native, 4) is base_result
    assert [call[0] for call in calls] == ["base"]


def test_unsupported_long_shape_selects_common_before_execution(route):
    state, calls, native, _, base_result, _, _ = route
    state.outcome = None
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert indexer.producer(native, 4) is base_result
    assert [call[0] for call in calls] == ["long", "base"]


def test_execution_error_never_retries(route):
    state, calls, native, _, _, _, _ = route
    state.outcome = RuntimeError("kernel failure")
    with (
        policy_context(SelectionPolicy(prefer="flagos", strict=False)),
        pytest.raises(RuntimeError, match="kernel failure"),
    ):
        indexer.producer(native, 4)
    assert [call[0] for call in calls] == ["long"]


def test_reference_policy_and_missing_operator_use_native(route):
    _, calls, native, native_result, _, _, funcs = route
    with policy_context(SelectionPolicy(prefer="reference")):
        assert indexer.compact(native, 4) is native_result
    assert [call[0] for call in calls] == ["native"]
    calls.clear()
    del funcs["dsv41_long_context_producer"]
    # The manager caches by policy, so use the other op with its own cache key.
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert indexer.producer(native, 5) is native_result
    assert [call[0] for call in calls] == ["native"]
