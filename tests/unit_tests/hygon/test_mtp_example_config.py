# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""HCU MTP configuration and strict-counter gates without a GPU or SGLang import."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/qwen3_6_27b_mtp_inference.py"


def load_example(monkeypatch, model="/models/Qwen3.6-27B", vendor="hcu", tp=None):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    if vendor == "hcu":
        torch.__hcu_version__ = "3.6"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("MODEL_PATH", model)
    # The example detects T-Head PPU from PPU_SDK; keep a PPU host from leaking in.
    monkeypatch.delenv("PPU_SDK", raising=False)
    if tp is None:
        monkeypatch.delenv("TP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TP_SIZE", str(tp))
    return runpy.run_path(str(EXAMPLE), run_name="hcu_mtp_cpu_test")


def install_engine(monkeypatch, mutation=None):
    captured = []

    # Match the actual0.5.18 field names. No **kwargs: removed graph arguments
    # fail at the Engine boundary rather than being swallowed by a mock.
    def engine(
        *,
        model_path,
        tp_size,
        mem_fraction_static,
        disable_cuda_graph,
        disable_overlap_schedule,
        trust_remote_code,
        disable_radix_cache,
        nnodes,
        pp_size,
        attention_backend,
        moe_runner_backend,
        speculative_moe_runner_backend,
        dtype,
        kv_cache_dtype,
        page_size,
        context_length,
        max_running_requests,
        chunked_prefill_size,
        disable_custom_all_reduce,
        cuda_graph_backend_prefill,
        cuda_graph_backend_decode,
        watchdog_timeout,
        tokenizer_path,
        speculative_algorithm=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        speculative_draft_model_path=None,
    ):
        actual = dict(locals())
        actual["cuda_graph_config"] = SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled"),
            decode=SimpleNamespace(backend="disabled"),
        )
        if mutation:
            mutation(actual)
        obj = SimpleNamespace(server_args=SimpleNamespace(**actual), shutdown_calls=0)

        def shutdown():
            obj.shutdown_calls += 1

        obj.shutdown = shutdown
        captured.append(obj)
        return obj

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = engine
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", module)
    return captured


@pytest.mark.parametrize("model", ["/models/Qwen3.6-27B", "/models/Qwen3.6-35B-A3B"])
@pytest.mark.parametrize("factory", ["_make_mtp_engine", "_make_baseline_engine"])
def test_hcu_factories_use_same_controlled_profile(monkeypatch, model, factory):
    ns = load_example(monkeypatch, model)
    captured = install_engine(monkeypatch)
    obj = ns[factory]()
    args = obj.server_args
    assert ns["TP_SIZE"] == args.tp_size == 2
    assert args.model_path == args.tokenizer_path == model
    assert args.nnodes == args.pp_size == 1
    assert args.dtype == args.kv_cache_dtype == "bfloat16"
    assert args.attention_backend == "hcu"
    assert args.moe_runner_backend == args.speculative_moe_runner_backend == "triton"
    assert args.mem_fraction_static == 0.70
    assert args.context_length == args.chunked_prefill_size == 4096
    assert args.page_size == 64 and args.max_running_requests == 16
    assert args.disable_cuda_graph and args.disable_overlap_schedule
    assert args.disable_radix_cache and args.disable_custom_all_reduce
    assert (
        args.cuda_graph_backend_prefill == args.cuda_graph_backend_decode == "disabled"
    )
    assert captured == [obj] and obj.shutdown_calls == 0
    if factory == "_make_mtp_engine":
        assert args.speculative_draft_model_path == model
        assert args.speculative_algorithm == "EAGLE"
        assert (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        ) == (3, 1, 4)
    else:
        assert args.speculative_algorithm is None


def test_hcu_forces_eager_without_old_constructor_options(monkeypatch):
    ns = load_example(monkeypatch)
    flags = SimpleNamespace(
        disable_cuda_graph=False,
        disable_piecewise_cuda_graph=False,
        disable_overlap_schedule=False,
    )
    ns["_enforce_hcu_mtp_runtime_mode"](flags)
    assert (
        flags.disable_cuda_graph
        and flags.disable_piecewise_cuda_graph
        and flags.disable_overlap_schedule
    )
    assert ns["_piecewise_graph_kwargs"](True) == {}
    assert "enable_breakable_cuda_graph" not in ns["_platform_engine_kwargs"]()


def test_hcu_rejects_uncontrolled_tp_before_constructing_engine(monkeypatch):
    ns = load_example(monkeypatch, tp=4)
    calls = install_engine(monkeypatch)
    with pytest.raises(ValueError, match="TP_SIZE=2"):
        ns["_make_mtp_engine"]()
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(max_running_requests=48),
        lambda a: a.update(tokenizer_path="/models/unrelated"),
        lambda a: setattr(a["cuda_graph_config"].decode, "backend", "full"),
    ],
)
def test_normalization_drift_fails_and_cleans_up_engine(monkeypatch, mutation):
    ns = load_example(monkeypatch)
    calls = install_engine(monkeypatch, mutation)
    with pytest.raises(RuntimeError):
        ns["_make_mtp_engine"]()
    assert len(calls) == 1 and calls[0].shutdown_calls == 1


def response():
    return {
        "text": "Actual output text",
        "meta_info": {
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "spec_verify_ct": 2,
            "spec_num_proposed_drafts": 6,
            "spec_num_correct_drafts": 3,
            "spec_accept_rate": 0.5,
            "spec_accept_length": 3.5,
        },
    }


def test_actual_metadata_and_speculative_ratios(monkeypatch):
    ns = load_example(monkeypatch)
    meta = ns["_generation_metadata"](response(), speculative=True)
    assert meta["completion_tokens"] == 7


@pytest.mark.parametrize(
    "key,value",
    [
        ("completion_tokens", None),
        ("completion_tokens", True),
        ("spec_verify_ct", 0),
        ("spec_num_proposed_drafts", 4),
        ("spec_num_correct_drafts", 7),
        ("spec_accept_rate", float("nan")),
        ("spec_accept_length", 2),
    ],
)
def test_missing_or_forged_actual_metadata_fails(monkeypatch, key, value):
    ns = load_example(monkeypatch)
    obj = response()
    obj["meta_info"][key] = value
    with pytest.raises(ValueError):
        ns["_generation_metadata"](obj, speculative=True)


def test_long_generation_uses_actual_tokens_and_never_word_count(monkeypatch):
    ns = load_example(monkeypatch)
    fn = ns["run_long_generation"]
    fn.__globals__["_text_prompt"] = lambda prompt: prompt
    engine = SimpleNamespace(generate=lambda **kwargs: response())
    _, tokens, _ = fn(engine, "test", max_tokens=512, expect_speculative=True)
    assert tokens == 7
    obj = response()
    del obj["meta_info"]["completion_tokens"]
    engine.generate = lambda **kwargs: obj
    with pytest.raises(ValueError):
        fn(engine, "test", expect_speculative=True)


@pytest.mark.parametrize("value", [None, float("inf"), float("nan"), True, 0, 5])
def test_acceptance_stats_cannot_be_missing_nonfinite_or_outside_window(
    monkeypatch, value
):
    ns = load_example(monkeypatch)
    with pytest.raises(ValueError):
        ns["_spec_accept_length"](
            {"internal_states": [{"avg_spec_accept_length": value}]}
        )


def test_acceptance_gate_keeps_lowest_scheduler_value(monkeypatch):
    ns = load_example(monkeypatch)
    assert (
        ns["_spec_accept_length"](
            {
                "internal_states": [
                    {"avg_spec_accept_length": 3.2},
                    {"avg_spec_accept_length": 1.9},
                ]
            }
        )
        == 1.9
    )
    # Existing main acceptance remains strictly>2 and does not skip missing stats.
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "ACCEPT_THRESHOLD = 2.0" in source
    assert 'print("    FAIL: stats not available")' in source
    assert "run_logprob_conformance(mtp_engine)" in source


def test_other_platform_defaults_are_unchanged(monkeypatch):
    ns = load_example(monkeypatch, vendor="cuda")
    assert ns["TP_SIZE"] == 1 and ns["_platform_engine_kwargs"]() == {}


@pytest.mark.parametrize("missing_spec_counter", [False, True])
def test_hcu_teacher_forced_conformance_requires_real_speculative_metadata(
    monkeypatch, missing_spec_counter
):
    ns = load_example(monkeypatch)
    check = ns["run_logprob_conformance"]
    check.__globals__["_text_prompt"] = lambda prompt: prompt
    prompt_ids = [1, 2, 3]
    output_ids = list(range(100, 132))

    class Engine:
        def generate(self, prompt=None, input_ids=None, sampling_params=None, **kwargs):
            if prompt is not None:
                assert sampling_params == {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                }
                result = {
                    "text": "actual generated tokens",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 32,
                        "spec_verify_ct": 8,
                        "spec_num_proposed_drafts": 24,
                        "spec_num_correct_drafts": 24,
                        "spec_accept_rate": 1.0,
                        "spec_accept_length": 4.0,
                        "input_token_logprobs": [(-0.2, i, None) for i in prompt_ids],
                        "output_token_logprobs": [(-0.1, i, None) for i in output_ids],
                        "output_top_logprobs": [
                            [(-0.1, i, None), (-1.0, 999, None)] for i in output_ids
                        ],
                    },
                }
                if missing_spec_counter:
                    del result["meta_info"]["spec_verify_ct"]
                return result
            assert input_ids == prompt_ids + output_ids
            assert sampling_params == {"temperature": 0, "max_new_tokens": 0}
            # Scoring legitimately generates zero tokens and has no spec counters.
            return {
                "meta_info": {
                    "prompt_tokens": 35,
                    "completion_tokens": 0,
                    "input_token_logprobs": [(-0.1, i, None) for i in input_ids],
                    "input_top_logprobs": [
                        [(-0.1, i, None), (-1.0, 999, None)] for i in input_ids
                    ],
                }
            }

    if missing_spec_counter:
        with pytest.raises(ValueError, match="spec_verify_ct"):
            check(Engine())
    else:
        report = check(Engine())
        assert report["passed"] is True and report["scored_tokens"] == 64


def child_autoload_value():
    """Read the actual inherited process environment, not only os.environ."""
    return subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD', '<unset>'))",
        ],
        text=True,
    ).strip()


@pytest.mark.parametrize("factory", ["_make_mtp_engine", "_make_baseline_engine"])
@pytest.mark.parametrize("drift", ["python_mapping", "native_environment"])
def test_hcu_engine_boundary_reasserts_autoload_after_environment_drift(
    monkeypatch, factory, drift
):
    ns = load_example(monkeypatch)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if drift == "python_mapping":
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    else:
        # os.putenv changes the environment inherited by a new child without
        # synchronizing Python's mapping. setdefault("...", "0") would miss it.
        os.putenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
        assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert child_autoload_value() == "1"
    boundary_values = []

    def observe_boundary(actual):
        boundary_values.append(
            (os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"), child_autoload_value())
        )

    calls = install_engine(monkeypatch, observe_boundary)
    ns[factory]()
    assert boundary_values == [("0", "0")]
    assert len(calls) == 1


def test_second_hcu_engine_reasserts_native_environment_after_first_engine(
    monkeypatch,
):
    ns = load_example(monkeypatch)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    boundary_values = []

    def observe_boundary(actual):
        boundary_values.append(
            (os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"), child_autoload_value())
        )

    calls = install_engine(monkeypatch, observe_boundary)
    first = ns["_make_mtp_engine"]()
    first.shutdown()
    # Controlled drift between independent factories reproduces the boundary
    # condition without attributing the external mutation to any dependency.
    os.putenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert child_autoload_value() == "1"
    second = ns["_make_baseline_engine"]()
    assert boundary_values == [("0", "0"), ("0", "0")]
    assert len(calls) == 2 and first.shutdown_calls == 1
    assert second.server_args.speculative_algorithm is None


@pytest.mark.parametrize("factory", ["_make_mtp_engine", "_make_baseline_engine"])
def test_non_hcu_factory_preserves_autoload_environment(monkeypatch, factory):
    ns = load_example(monkeypatch, vendor="cuda")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    observed = []
    marker = object()

    def engine(**kwargs):
        observed.append(
            (os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"), child_autoload_value())
        )
        return marker

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = engine
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", module)
    assert ns[factory]() is marker
    assert observed == [("1", "1")]
    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "1"
