# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise Iluvatar example setup without allocating a model or a GPU."""

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
    captured = _capture_engine(monkeypatch)
    namespace = _load_example(monkeypatch, name)
    with pytest.raises(CapturedEngine):
        namespace[factory]()
    assert captured["tp_size"] == 1


@pytest.mark.parametrize("concurrency", [1, 8, 16, 32])
def test_corex_moe_graph_limit_covers_concurrent_requests(monkeypatch, concurrency):
    _cuda_platform(monkeypatch, corex=True)
    monkeypatch.setenv("CONCURRENT_N", str(concurrency))
    captured = _capture_engine(monkeypatch)
    namespace = _load_example(monkeypatch, "qwen3_6_35b_a3b_concurrent.py")
    with pytest.raises(CapturedEngine):
        namespace["_make_engine"]()
    assert captured["cuda_graph_max_bs_decode"] == concurrency


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
