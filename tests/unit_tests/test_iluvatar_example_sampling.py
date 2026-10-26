# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture actual example generation arguments without importing a GPU runtime."""

import asyncio
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = [
    f"qwen3_6_{model}_{kind}.py"
    for model in ("27b", "35b_a3b")
    for kind in ("offline_inference", "concurrent")
]
_ENV = "FL_CONCURRENT_VL_MAX_TOKENS"


def _load(monkeypatch, name, vendor, budget):
    # Replace only the hardware probe. The real example code and runners execute.
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    if vendor == "corex":
        torch.corex = SimpleNamespace()
    elif vendor == "hygon":
        torch.__hcu_version__ = "3.6"
    elif vendor != "cuda":
        attr = {"musa": "musa", "ascend": "npu", "tsingmicro": "txda"}[vendor]
        setattr(torch, attr, SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    stub = ModuleType(
        "sglang_fl.dispatch.backends.vendor.tsingmicro.patches.platform_stubs"
    )
    stub.patch = lambda: None
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    # Track setdefault targets in other-platform example initialization.
    for key in (
        "GLOO_SOCKET_IFNAME",
        "HCCL_IF_BASE_PORT",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
        "HCCL_BUFFSIZE",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
        "SGLANG_FL_TIMER_ENABLE",
        "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_REQ_RUNNING_TIMEOUT",
    ):
        monkeypatch.setenv(key, "sampling-config-test")
    for key in ("TP_SIZE", "ATTENTION_BACKEND", "IMAGE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CONCURRENT_N", "16")
    monkeypatch.setenv("MAX_TOKENS", "97")
    monkeypatch.setenv("DISABLE_CUDA_GRAPH", "1")
    if budget is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, budget)
    return runpy.run_path(
        str(_ROOT / "examples" / name), run_name="sampling_config_test"
    )


def _capture_generation(monkeypatch, namespace, name):
    calls = []
    engines = []

    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loop = asyncio.new_event_loop()
            self.closed = False
            engines.append(self)

        def generate(self, *, prompt, sampling_params, image_data=None):
            calls.append({"sampling": dict(sampling_params), "image_data": image_data})
            # Output is deliberately not a semantic-pass fixture. This test only
            # inspects arguments sent by the unmodified example runner.
            return {
                "text": "argument-capture-only",
                "meta_info": {"completion_tokens": 1},
            }

        async def async_generate(self, **kwargs):
            return self.generate(**kwargs)

        def shutdown(self):
            self.closed = True
            self.loop.close()

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    factory = namespace.get("_make_engine", namespace.get("run_engine"))
    globals_ = factory.__globals__
    # Prompt templating is irrelevant to token-budget forwarding; avoid loading
    # transformers or a checkpoint while retaining actual image URI construction.
    monkeypatch.setitem(globals_, "_text_prompt", lambda question: question)
    monkeypatch.setitem(globals_, "_vl_prompt", lambda question, uri: question)
    if "offline" in name:
        namespace["run_engine"]()
        assert len(calls) == 2 + 4
        assert engines[0].closed
    else:
        engine = namespace["_make_engine"]()
        try:
            namespace["run_vl_concurrent"](engine)
            vl_calls = calls[:]
            assert len(vl_calls) == 16 and all(c["image_data"] for c in vl_calls)
            namespace["run_mixed_concurrent"](engine)
            mixed_calls = calls[16:]
            assert len(mixed_calls) == 16
            assert sum(bool(c["image_data"]) for c in mixed_calls) == 8
        finally:
            engine.shutdown()
    assert len(engines) == 1
    return engines[0].kwargs, calls


@pytest.mark.parametrize("name", _EXAMPLES)
@pytest.mark.parametrize("budget,expected", [(None, 256), ("1024", 1024)])
def test_corex_budget_reaches_all_original_vl_generation_calls(
    monkeypatch, name, budget, expected
):
    namespace = _load(monkeypatch, name, "corex", budget)
    kwargs, calls = _capture_generation(monkeypatch, namespace, name)
    assert kwargs["tp_size"] == 4
    assert kwargs["disable_cuda_graph"] is True
    assert {c["sampling"]["max_new_tokens"] for c in calls if c["image_data"]} == {
        expected
    }
    assert {c["sampling"]["max_new_tokens"] for c in calls if not c["image_data"]} == {
        97
    }
    assert all(c["sampling"]["temperature"] == 0 for c in calls)


@pytest.mark.parametrize("name", _EXAMPLES)
@pytest.mark.parametrize("vendor", ["cuda", "hygon", "musa", "ascend", "tsingmicro"])
@pytest.mark.parametrize("budget", ["256", "not-an-integer"])
def test_other_backends_keep_64_for_all_vl_generation_calls(
    monkeypatch, name, vendor, budget
):
    namespace = _load(monkeypatch, name, vendor, budget)
    _, calls = _capture_generation(monkeypatch, namespace, name)
    assert {c["sampling"]["max_new_tokens"] for c in calls if c["image_data"]} == {64}
    assert {c["sampling"]["max_new_tokens"] for c in calls if not c["image_data"]} == {
        97
    }


@pytest.mark.parametrize("name", _EXAMPLES)
@pytest.mark.parametrize("budget", ["0", "-1", "not-an-integer"])
def test_corex_invalid_budget_is_rejected_before_engine(monkeypatch, name, budget):
    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = lambda **kwargs: pytest.fail(
        "Invalid VL budget must not create an Engine"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError):
        _load(monkeypatch, name, "corex", budget)


_CONCURRENT = [f"qwen3_6_{model}_concurrent.py" for model in ("27b", "35b_a3b")]


@pytest.mark.parametrize("name", _CONCURRENT)
def test_corex_flush_waits_for_scheduler_acknowledgement(monkeypatch, name):
    namespace = _load(monkeypatch, name, "corex", None)
    flush = namespace["_flush_between_modes"]
    replies = [SimpleNamespace(success=False), SimpleNamespace(success=True)]
    calls = []
    sleeps = []
    clock = SimpleNamespace(monotonic=lambda: 0.0, sleep=sleeps.append)
    monkeypatch.setitem(flush.__globals__, "time", clock)

    def request():
        reply = replies[len(calls)]
        calls.append(reply)
        return reply

    assert flush(SimpleNamespace(flush_cache=request)) is replies[1]
    assert calls == replies
    assert sleeps == [0.1]


@pytest.mark.parametrize("name", _CONCURRENT)
def test_corex_busy_flush_fails_when_retry_budget_expires(monkeypatch, name):
    namespace = _load(monkeypatch, name, "corex", None)
    flush = namespace["_flush_between_modes"]
    ticks = iter([0.0, 31.0])
    clock = SimpleNamespace(
        monotonic=lambda: next(ticks),
        sleep=lambda _: pytest.fail("Expired flush must not wait or proceed"),
    )
    monkeypatch.setitem(flush.__globals__, "time", clock)
    engine = SimpleNamespace(flush_cache=lambda: SimpleNamespace(success=False))
    with pytest.raises(RuntimeError, match="retry budget"):
        flush(engine)


@pytest.mark.parametrize("name", _CONCURRENT)
@pytest.mark.parametrize("reply", [None, SimpleNamespace(), SimpleNamespace(success=1)])
def test_corex_flush_rejects_unacknowledged_result(monkeypatch, name, reply):
    namespace = _load(monkeypatch, name, "corex", None)
    with pytest.raises(RuntimeError, match="explicit success state"):
        namespace["_flush_between_modes"](SimpleNamespace(flush_cache=lambda: reply))


@pytest.mark.parametrize("name", _CONCURRENT)
@pytest.mark.parametrize("vendor", ["cuda", "hygon", "musa", "ascend", "tsingmicro"])
def test_other_backends_retain_single_flush_call(monkeypatch, name, vendor):
    namespace = _load(monkeypatch, name, vendor, None)
    flush = namespace["_flush_between_modes"]
    reply = SimpleNamespace(success=False)
    calls = []

    def request():
        calls.append(True)
        return reply

    assert flush(SimpleNamespace(flush_cache=request)) is reply
    assert calls == [True]


@pytest.mark.parametrize("name", _CONCURRENT)
def test_corex_flush_does_not_retry_after_sleep_exhausts_budget(monkeypatch, name):
    namespace = _load(monkeypatch, name, "corex", None)
    flush = namespace["_flush_between_modes"]
    ticks = iter([0.0, 29.95, 30.01])
    sleeps = []
    calls = []
    monkeypatch.setitem(
        flush.__globals__,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks), sleep=sleeps.append),
    )

    def request():
        calls.append(True)
        if len(calls) > 1:
            pytest.fail("An RPC must not start after the retry budget expires")
        return SimpleNamespace(success=False)

    with pytest.raises(RuntimeError, match="retry budget"):
        flush(SimpleNamespace(flush_cache=request))
    assert calls == [True]
    assert sleeps == [pytest.approx(0.05)]
