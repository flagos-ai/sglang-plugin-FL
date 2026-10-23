# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise Iluvatar example setup without allocating a model or a GPU."""

from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_SINGLE_CASES = [
    ("qwen3_6_27b_offline_inference.py", "run_engine"),
    ("qwen3_6_35b_a3b_offline_inference.py", "run_engine"),
    ("qwen3_6_27b_concurrent.py", "_make_engine"),
    ("qwen3_6_35b_a3b_concurrent.py", "_make_engine"),
]
_OFFLINE = [name for name, factory in _SINGLE_CASES if factory == "run_engine"]
_MULTINODE = ["qwen3_6_27b_multinode.py", "qwen3_6_35b_a3b_multinode.py"]


class CapturedEngine(Exception):
    pass


def _capture_engine(monkeypatch):
    captured = {}

    def engine(**kwargs):
        captured.update(kwargs)
        raise CapturedEngine

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return captured


def _load_example(monkeypatch, name):
    monkeypatch.syspath_prepend(str(_ROOT / "examples"))
    return runpy.run_path(str(_ROOT / "examples" / name))


def _cuda_platform(monkeypatch, *, corex):
    for name in ("musa", "npu", "txda"):
        monkeypatch.setattr(
            torch, name, SimpleNamespace(is_available=lambda: False), raising=False
        )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delattr(torch, "__hcu_version__", raising=False)
    if corex:
        monkeypatch.setattr(torch, "corex", SimpleNamespace(), raising=False)
    else:
        monkeypatch.delattr(torch, "corex", raising=False)


@pytest.mark.parametrize("disabled", [None, "0", "1"])
@pytest.mark.parametrize("name,factory", _SINGLE_CASES)
def test_acceptance_graph_switch_reaches_engine(monkeypatch, disabled, name, factory):
    captured = _capture_engine(monkeypatch)
    if disabled is None:
        monkeypatch.delenv("DISABLE_CUDA_GRAPH", raising=False)
    else:
        monkeypatch.setenv("DISABLE_CUDA_GRAPH", disabled)
    namespace = _load_example(monkeypatch, name)
    with pytest.raises(CapturedEngine):
        namespace[factory]()
    assert captured["disable_cuda_graph"] is (disabled != "0")


@pytest.mark.parametrize("tp", [None, "2"])
@pytest.mark.parametrize("name,factory", _SINGLE_CASES)
def test_corex_default_tp_and_explicit_override(monkeypatch, tp, name, factory):
    _cuda_platform(monkeypatch, corex=True)
    if tp is None:
        monkeypatch.delenv("TP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TP_SIZE", tp)
    captured = _capture_engine(monkeypatch)
    namespace = _load_example(monkeypatch, name)
    with pytest.raises(CapturedEngine):
        namespace[factory]()
    assert captured["tp_size"] == (4 if tp is None else 2)


@pytest.mark.parametrize("name,factory", _SINGLE_CASES)
def test_other_cuda_default_tp_stays_one(monkeypatch, name, factory):
    _cuda_platform(monkeypatch, corex=False)
    monkeypatch.delenv("TP_SIZE", raising=False)
    monkeypatch.setenv("CONCURRENT_N", "64")
    captured = _capture_engine(monkeypatch)
    namespace = _load_example(monkeypatch, name)
    with pytest.raises(CapturedEngine):
        namespace[factory]()
    assert captured["tp_size"] == 1
    assert "cuda_graph_max_bs_decode" not in captured


@pytest.mark.parametrize("concurrency", [1, 8, 16, 32, 64])
@pytest.mark.parametrize(
    "name", ["qwen3_6_27b_concurrent.py", "qwen3_6_35b_a3b_concurrent.py"]
)
def test_corex_graph_limit_covers_concurrent_requests(monkeypatch, concurrency, name):
    _cuda_platform(monkeypatch, corex=True)
    monkeypatch.setenv("CONCURRENT_N", str(concurrency))
    _install_current_server_args_engine(monkeypatch)
    namespace = _load_example(monkeypatch, name)
    engine = namespace["_make_engine"]()
    assert engine.server_args.cuda_graph_max_bs_decode == concurrency


@dataclass
class _ServerArgs0518:
    """Strict subset of the upstream v0.5.18 Python constructor used here.

    Source: sgl-project/sglang v0.5.18 python/sglang/srt/server_args.py.
    The old --cuda-graph-max-bs CLI alias targets cuda_graph_max_bs_decode;
    cuda_graph_max_bs is not a ServerArgs constructor field. No **kwargs are
    accepted here, so an obsolete Python argument cannot pass as a capture.
    """

    model_path: str
    tp_size: int = 1
    mem_fraction_static: float = 0.8
    disable_cuda_graph: bool = False
    disable_overlap_schedule: bool = False
    trust_remote_code: bool = False
    disable_radix_cache: bool = False
    attention_backend: str | None = None
    watchdog_timeout: float = 300
    cuda_graph_max_bs_decode: int | None = None
    disable_prefill_cuda_graph: bool = False
    speculative_algorithm: str | None = None
    speculative_num_steps: int | None = None
    speculative_eagle_topk: int | None = None
    speculative_num_draft_tokens: int | None = None


def _install_current_server_args_engine(monkeypatch):
    class Engine:
        def __init__(self, **kwargs):
            # Match upstream Engine -> self.server_args_class(**kwargs).
            self.server_args = _ServerArgs0518(**kwargs)

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = _ServerArgs0518
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)


def _load_mtp_example(monkeypatch, vendor="iluvatar"):
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    # Execute the real script with an explicit device identity. This creates no
    # accelerator context and does not depend on the machine running pytest.
    probed_torch = ModuleType("torch")
    probed_torch.cuda = SimpleNamespace(is_available=lambda: True)
    if vendor == "iluvatar":
        probed_torch.corex = SimpleNamespace()
    elif vendor == "hygon":
        probed_torch.__hcu_version__ = "3.6"
    elif vendor != "cuda":
        attr = {"musa": "musa", "ascend": "npu", "tsingmicro": "txda"}[vendor]
        setattr(probed_torch, attr, SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", probed_torch)
    # Track the example's setdefault targets so platform setup cannot leak into
    # later tests when exercising Ascend/TXDA identities.
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
        monkeypatch.setenv(key, os.environ.get(key, "test"))
    stubs = ModuleType(
        "sglang_fl.dispatch.backends.vendor.tsingmicro.patches.platform_stubs"
    )
    stubs.patch = lambda: None
    monkeypatch.setitem(sys.modules, stubs.__name__, stubs)
    return _load_example(monkeypatch, "qwen3_6_27b_mtp_inference.py")


@pytest.mark.parametrize("factory", ["_make_mtp_engine", "_make_baseline_engine"])
@pytest.mark.parametrize("tp", [None, "2"])
@pytest.mark.parametrize("eager", [False, True])
def test_corex_mtp_factories_use_current_server_args(monkeypatch, factory, tp, eager):
    if tp is None:
        monkeypatch.delenv("TP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TP_SIZE", tp)
    _install_current_server_args_engine(monkeypatch)
    namespace = _load_mtp_example(monkeypatch)
    engine = namespace[factory](
        disable_cuda_graph=eager,
        disable_piecewise_cuda_graph=eager,
        disable_overlap_schedule=eager,
    )
    args = engine.server_args
    assert args.tp_size == (4 if tp is None else 2)
    assert args.attention_backend == "triton"
    assert args.watchdog_timeout == 3600
    assert args.cuda_graph_max_bs_decode == 16
    assert args.disable_cuda_graph is eager
    assert args.disable_prefill_cuda_graph is eager
    assert args.disable_overlap_schedule is eager
    if factory == "_make_mtp_engine":
        assert args.speculative_algorithm == "EAGLE"
        assert (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        ) == (3, 1, 4)
    else:
        assert args.speculative_algorithm is None


@pytest.mark.parametrize("factory", ["_make_mtp_engine", "_make_baseline_engine"])
def test_legacy_corex_mtp_keyword_is_rejected_by_current_server_args(
    monkeypatch, factory
):
    _install_current_server_args_engine(monkeypatch)
    namespace = _load_mtp_example(monkeypatch)
    platform_kwargs = namespace["_platform_engine_kwargs"]

    def old_corex_kwargs():
        kwargs = platform_kwargs()
        kwargs["cuda_graph_max_bs"] = kwargs.pop("cuda_graph_max_bs_decode", 16)
        return kwargs

    monkeypatch.setitem(
        namespace[factory].__globals__, "_platform_engine_kwargs", old_corex_kwargs
    )
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'cuda_graph_max_bs'"
    ):
        namespace[factory]()


@pytest.mark.parametrize(
    "vendor,tp",
    [("cuda", 1), ("musa", 1), ("ascend", 4), ("tsingmicro", 4), ("hygon", 1)],
)
def test_mtp_other_platform_defaults_remain_unchanged(monkeypatch, vendor, tp):
    monkeypatch.delenv("TP_SIZE", raising=False)
    namespace = _load_mtp_example(monkeypatch, vendor)
    assert namespace["TP_SIZE"] == tp
    assert "cuda_graph_max_bs_decode" not in namespace["_platform_engine_kwargs"]()


@pytest.mark.parametrize("label", [None, "Qwen3.6-35B-A3B"])
def test_mtp_main_uses_optional_model_label_without_changing_default(
    monkeypatch, tmp_path, capsys, label
):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["mtp.py"])
    namespace = _load_mtp_example(monkeypatch)

    def stop_before_model(**kwargs):
        raise CapturedEngine

    monkeypatch.setitem(
        namespace["main"].__globals__, "_make_mtp_engine", stop_before_model
    )
    with pytest.raises(CapturedEngine):
        if label is None:
            namespace["main"]()
        else:
            namespace["main"](model_label=label)
    expected = label or "Qwen3.6-27B"
    assert (
        f"{expected} MTP (Speculative Decoding) Validation" in capsys.readouterr().out
    )


@pytest.mark.parametrize("name", _OFFLINE)
def test_offline_missing_images_fail_before_engine_creation(
    monkeypatch, tmp_path, name
):
    monkeypatch.setenv("IMAGE_DIR", str(tmp_path))
    (tmp_path / "red_square.jpg").write_bytes(b"fixture")
    captured = _capture_engine(monkeypatch)
    namespace = _load_example(monkeypatch, name)
    with pytest.raises(FileNotFoundError, match="Required test images.*cat.jpg"):
        namespace["run_engine"]()
    assert captured == {}


@pytest.mark.parametrize("name", _OFFLINE)
def test_offline_missing_vl_outputs_cannot_pass_validation(monkeypatch, name):
    namespace = _load_example(monkeypatch, name)
    text = [namespace["TEXT_EXPECTED"][prompt] for prompt in namespace["TEXT_PROMPTS"]]
    with pytest.raises(AssertionError, match="Missing output for required image"):
        namespace["validate"](text, [None] * len(namespace["VL_CASES"]))


@pytest.mark.parametrize("entry", ["run_master", "run_tests"])
@pytest.mark.parametrize("name", _MULTINODE)
def test_multinode_missing_images_fail_before_server_or_requests(
    monkeypatch, tmp_path, name, entry
):
    monkeypatch.setenv("IMAGE_DIR", str(tmp_path))
    (tmp_path / "red_square.jpg").write_bytes(b"fixture")
    namespace = _load_example(monkeypatch, name)

    def unexpected(*args, **kwargs):
        raise AssertionError("missing images must fail before model or HTTP work")

    monkeypatch.setattr(namespace["subprocess"], "Popen", unexpected)
    monkeypatch.setitem(namespace["run_tests"].__globals__, "chat_request", unexpected)
    with pytest.raises(FileNotFoundError, match="Required test images.*cat.jpg"):
        if entry == "run_master":
            namespace[entry](SimpleNamespace())
        else:
            namespace[entry](30000, 4, 2)


@pytest.mark.parametrize("concurrency", ["0", "-1"])
@pytest.mark.parametrize(
    "name", ["qwen3_6_27b_concurrent.py", "qwen3_6_35b_a3b_concurrent.py"]
)
def test_examples_reject_nonpositive_request_counts(monkeypatch, name, concurrency):
    monkeypatch.setenv("CONCURRENT_N", concurrency)
    with pytest.raises(ValueError, match="CONCURRENT_N must be a positive integer"):
        _load_example(monkeypatch, name)


def _bash():
    if os.name == "nt":
        git_bash = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Git/bin/bash.exe"
        )
        if git_bash.is_file():
            return str(git_bash)
    found = shutil.which("bash")
    if found:
        return found
    pytest.skip("Bash is required to exercise the acceptance script")


def _shell_path(path):
    value = path.resolve().as_posix()
    if os.name == "nt":
        return "/" + value[0].lower() + value[2:]
    return value


def _acceptance_harness(tmp_path, profile="vendor"):
    script = tmp_path / "scripts/iluvatar/run_single_node_acceptance.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        (_ROOT / "scripts/iluvatar/run_single_node_acceptance.sh").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text(
        '#!/bin/bash\nprintf \'%s\\t%s\\n\' "${DISABLE_CUDA_GRAPH:-unset}" "$*" >> "$CALL_LOG"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    calls = tmp_path / "calls.log"
    logs = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        PATH=f"{_shell_path(binaries)}:/usr/bin:/bin",
        CALL_LOG=_shell_path(calls),
        OUTPUT_DIR=_shell_path(logs),
        FUSED_PROFILE=profile,
        TP_SIZE="1",
        MAX_TOKENS="4",
        CONCURRENT_N="8",
        SGLANG_FL_DIST_BACKEND="nccl",
    )
    env.pop("DISABLE_CUDA_GRAPH", None)
    return script, calls, logs, env


def _run_acceptance(script, tmp_path, env, *, check):
    return subprocess.run(
        [_bash(), "--noprofile", "--norc", _shell_path(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=30,
    )


@pytest.mark.parametrize("profile", ["vendor", "flagos"])
def test_acceptance_runs_both_models_and_concurrency_with_each_graph_mode(
    tmp_path, profile
):
    script, calls, logs, env = _acceptance_harness(tmp_path, profile)
    _run_acceptance(script, tmp_path, env, check=True)
    invocations = [line.split("\t", 1) for line in calls.read_text().splitlines()]
    assert f"--fused-profile {profile}" in invocations[0][1]
    expected = []
    for model in ("27b", "35b_a3b"):
        for disabled in ("1", "0"):
            expected.extend(
                [
                    [disabled, f"examples/qwen3_6_{model}_offline_inference.py"],
                    [disabled, f"examples/qwen3_6_{model}_concurrent.py --mode all"],
                ]
            )
            assert (logs / f"{model}_offline_graph_disabled_{disabled}.log").is_file()
            assert (
                logs / f"{model}_concurrent_graph_disabled_{disabled}.log"
            ).is_file()
    assert invocations[1:] == expected


@pytest.mark.parametrize(
    "name,value",
    [
        ("CONCURRENT_N", "0"),
        ("CONCURRENT_N", "1"),
        ("TP_SIZE", "0"),
        ("MAX_TOKENS", "0"),
    ],
)
def test_acceptance_rejects_invalid_workload_before_running_python(
    tmp_path, name, value
):
    script, calls, _, env = _acceptance_harness(tmp_path)
    env[name] = value
    result = _run_acceptance(script, tmp_path, env, check=False)
    assert result.returncode == 2
    assert name in result.stderr
    assert not calls.exists()


def _iluvatar_35b_mtp_entry(monkeypatch):
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    _cuda_platform(monkeypatch, corex=True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("GEMS_VENDOR", "iluvatar")
    monkeypatch.delenv("TP_SIZE", raising=False)
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    return _load_example(monkeypatch, "qwen3_6_35b_a3b_mtp_inference.py")["main"]


def _mtp_moe_config(tmp_path, contents=None):
    if contents is None:
        contents = (
            '{"architectures":["Qwen3_5MoeForConditionalGeneration"],'
            '"text_config":{"mtp_num_hidden_layers":1}}'
        )
    (tmp_path / "config.json").write_text(contents, encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("tp_size", [None, "4"])
@pytest.mark.parametrize("return_code", [0, 1])
def test_iluvatar_35b_mtp_requires_eager_baseline_and_propagates_result(
    monkeypatch, tmp_path, tp_size, return_code
):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    model_path = _mtp_moe_config(tmp_path)
    monkeypatch.setenv("MODEL_PATH", model_path)
    monkeypatch.setenv("DISABLE_CUDA_GRAPH", "0")
    if tp_size is not None:
        monkeypatch.setenv("TP_SIZE", tp_size)
    harness = SimpleNamespace()
    calls = []

    def validate(*, model_label):
        calls.append((model_label, list(sys.argv)))
        assert harness.MODEL_PATH == model_path
        assert harness.TP_SIZE == 4
        assert harness.MAX_TOKENS == 256
        return return_code

    harness.main = validate
    monkeypatch.setitem(entry.__globals__, "_load_harness", lambda: harness)
    original_argv = sys.argv
    assert entry([]) == return_code
    assert sys.argv is original_argv
    assert len(calls) == 1
    label, arguments = calls[0]
    assert label == "Qwen3.6-35B-A3B"
    assert "--skip-baseline" not in arguments
    for flag in (
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule",
    ):
        assert flag in arguments
    assert arguments[arguments.index("--max-tokens") + 1] == "256"


def test_iluvatar_35b_mtp_sets_model_default_and_restores_argv_on_error(monkeypatch):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    checkpoints = []
    monkeypatch.setitem(entry.__globals__, "_check_checkpoint", checkpoints.append)
    harness = SimpleNamespace()

    def failed_validation(*, model_label):
        assert model_label == "Qwen3.6-35B-A3B"
        assert harness.MODEL_PATH == "/models/Qwen3.6-35B-A3B"
        assert harness.MAX_TOKENS == 96
        raise RuntimeError("actual validation failed")

    harness.main = failed_validation
    monkeypatch.setitem(entry.__globals__, "_load_harness", lambda: harness)
    original_argv = sys.argv
    with pytest.raises(RuntimeError, match="actual validation failed"):
        entry(["--max-tokens", "96"])
    assert checkpoints == ["/models/Qwen3.6-35B-A3B"]
    assert sys.argv is original_argv


@pytest.mark.parametrize("tp_size", ["1", "2", "8", "not-an-integer"])
def test_iluvatar_35b_mtp_rejects_non_tp4_before_loading_harness(monkeypatch, tp_size):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    monkeypatch.setenv("TP_SIZE", tp_size)

    def unexpected():
        raise AssertionError("invalid settings must fail before harness loading")

    monkeypatch.setitem(entry.__globals__, "_load_harness", unexpected)
    with pytest.raises(SystemExit) as error:
        entry([])
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [["--skip-baseline"], ["--max-tokens", "0"]])
def test_iluvatar_35b_mtp_rejects_bypass_and_empty_workload(monkeypatch, arguments):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    with pytest.raises(SystemExit) as error:
        entry(arguments)
    assert error.value.code == 2


def test_iluvatar_35b_mtp_help_requires_no_hardware_or_harness(monkeypatch):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)

    def unexpected():
        raise AssertionError("help must not load hardware or the harness")

    monkeypatch.setitem(entry.__globals__, "_require_corex", unexpected)
    monkeypatch.setitem(entry.__globals__, "_load_harness", unexpected)
    with pytest.raises(SystemExit) as error:
        entry(["--help"])
    assert error.value.code == 0


@pytest.mark.parametrize(
    "contents,exception",
    [
        ("invalid json", ValueError),
        ('{"architectures":["Qwen3_5ForConditionalGeneration"]}', ValueError),
        ('{"architectures":"Qwen3_5MoeForConditionalGeneration"}', ValueError),
        ('{"architectures":["Qwen3_5MoeForConditionalGeneration"]}', ValueError),
        (
            '{"architectures":["Qwen3_5MoeForConditionalGeneration"],"text_config":{}}',
            ValueError,
        ),
        (
            '{"architectures":["Qwen3_5MoeForConditionalGeneration"],'
            '"text_config":{"mtp_num_hidden_layers":0}}',
            ValueError,
        ),
    ],
)
def test_iluvatar_35b_mtp_rejects_bad_checkpoint_before_loading_harness(
    monkeypatch, tmp_path, contents, exception
):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", _mtp_moe_config(tmp_path, contents))

    def unexpected():
        raise AssertionError("invalid checkpoint must fail before harness loading")

    monkeypatch.setitem(entry.__globals__, "_load_harness", unexpected)
    with pytest.raises(exception):
        entry([])


def test_iluvatar_35b_mtp_rejects_missing_config_before_loading_harness(
    monkeypatch, tmp_path
):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        entry([])


@pytest.mark.parametrize("mismatch", ["vendor", "corex", "devices"])
def test_iluvatar_35b_mtp_rejects_other_runtime_before_loading_harness(
    monkeypatch, mismatch
):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    if mismatch == "vendor":
        monkeypatch.setenv("GEMS_VENDOR", "nvidia")
    elif mismatch == "corex":
        monkeypatch.delattr(torch, "corex")
    else:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    with pytest.raises(RuntimeError):
        entry([])


def test_iluvatar_35b_mtp_loads_existing_strict_validation_harness(monkeypatch):
    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    harness = entry.__globals__["_load_harness"]()
    numeric_prompt = next(p for p in harness.PROMPTS if p.get("expected_number") == 50)
    assert harness._validate_output(numeric_prompt, "50")[0] is True
    assert harness._validate_output(numeric_prompt, "950")[0] is False
    assert callable(harness.run_logprob_conformance)
    assert callable(harness._make_mtp_engine)
    assert callable(harness._make_baseline_engine)


def _real_corex_mtp_main_before_engine(monkeypatch, tmp_path, model):
    """Run both entrypoints through the actual common main, stopping at Engine."""
    monkeypatch.setattr(sys, "argv", ["mtp.py"])
    if model == "27b":
        monkeypatch.setenv("MODEL_PATH", str(tmp_path))
        entry = _load_mtp_example(monkeypatch)["main"]
        return entry, entry.__globals__

    entry = _iluvatar_35b_mtp_entry(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", _mtp_moe_config(tmp_path))
    harness = entry.__globals__["_load_harness"]()
    monkeypatch.setitem(entry.__globals__, "_load_harness", lambda: harness)
    return lambda: entry([]), harness.main.__globals__


@pytest.mark.parametrize("model", ["27b", "35b_a3b"])
@pytest.mark.parametrize("value", ["2", "nan", "inf", "-inf", "bad"])
def test_corex_mtp_rejects_simulated_or_invalid_acceptance_before_engine(
    monkeypatch, tmp_path, model, value
):
    entry, namespace = _real_corex_mtp_main_before_engine(monkeypatch, tmp_path, model)
    monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", value)

    def unexpected(**kwargs):
        raise AssertionError("synthetic acceptance must fail before Engine creation")

    monkeypatch.setitem(namespace, "_make_mtp_engine", unexpected)
    with pytest.raises(SystemExit) as error:
        entry()
    assert error.value.code == 2


@pytest.mark.parametrize("model", ["27b", "35b_a3b"])
@pytest.mark.parametrize("value", [None, "0", "-1"])
def test_corex_mtp_allows_upstream_disabled_acceptance_values(
    monkeypatch, tmp_path, model, value
):
    entry, namespace = _real_corex_mtp_main_before_engine(monkeypatch, tmp_path, model)
    if value is not None:
        monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", value)

    def stop_at_engine(**kwargs):
        raise CapturedEngine

    monkeypatch.setitem(namespace, "_make_mtp_engine", stop_at_engine)
    with pytest.raises(CapturedEngine):
        entry()


@pytest.mark.parametrize("vendor", ["cuda", "musa", "ascend", "tsingmicro", "hygon"])
def test_mtp_acceptance_preflight_remains_corex_only(monkeypatch, tmp_path, vendor):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["mtp.py"])
    namespace = _load_mtp_example(monkeypatch, vendor)
    monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "2")

    def stop_at_engine(**kwargs):
        raise CapturedEngine

    monkeypatch.setitem(
        namespace["main"].__globals__, "_make_mtp_engine", stop_at_engine
    )
    with pytest.raises(CapturedEngine):
        namespace["main"]()
