# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise Hygon example configuration without initializing a GPU or server."""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SINGLE_NODE = [
    f"qwen3_6_{model}_{kind}.py"
    for model in ("27b", "35b_a3b")
    for kind in ("offline_inference", "concurrent")
]
_MULTINODE = [f"qwen3_6_{model}_multinode.py" for model in ("27b", "35b_a3b")]


def _load_example(monkeypatch, name, vendor="hygon"):
    # Only the example's hardware probe is stubbed; execute its actual module
    # configuration and function definitions, without invoking __main__.
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    if vendor == "hygon":
        torch.__hcu_version__ = "3.6"
    elif vendor == "musa":
        torch.musa = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    worker = ModuleType("_multinode_worker")
    worker.run_sglang_worker = lambda *args: pytest.fail("Must not launch a worker")
    monkeypatch.setitem(sys.modules, "_multinode_worker", worker)
    for name_to_clear in ("ATTENTION_BACKEND", "MODEL_PATH", "TP_SIZE", "IMAGE_DIR"):
        monkeypatch.delenv(name_to_clear, raising=False)
    return runpy.run_path(str(_ROOT / "examples" / name), run_name="hygon_config_test")


@pytest.mark.parametrize("name", _SINGLE_NODE)
def test_hygon_engine_uses_current_prefill_option_and_keeps_eager_decode(
    monkeypatch, name
):
    namespace = _load_example(monkeypatch, name)
    captured = {}

    class CapturedEngine(Exception):
        pass

    # Match the current ServerArgs constructor field names. In particular,
    # enable_breakable_cuda_graph was removed from the Python API in 0.5.18.
    def engine(
        *,
        model_path,
        tp_size,
        mem_fraction_static,
        disable_cuda_graph,
        dtype,
        kv_cache_dtype,
        page_size,
        disable_radix_cache,
        cuda_graph_backend_prefill,
        trust_remote_code,
    ):
        captured.update(
            model_path=model_path,
            tp_size=tp_size,
            disable_cuda_graph=disable_cuda_graph,
            cuda_graph_backend_prefill=cuda_graph_backend_prefill,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            page_size=page_size,
            disable_radix_cache=disable_radix_cache,
        )
        # Offline run_engine continues directly into model inference, so stop
        # after the actual Engine call has accepted and captured its kwargs.
        raise CapturedEngine

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = engine
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", module)
    factory = namespace.get("_make_engine", namespace.get("run_engine"))
    with pytest.raises(CapturedEngine):
        factory()
    assert captured["disable_cuda_graph"] is True
    assert captured["cuda_graph_backend_prefill"] == "disabled"
    assert captured["page_size"] == 64
    assert captured["disable_radix_cache"] is True
    assert captured["dtype"] == captured["kv_cache_dtype"] == "bfloat16"
    assert captured["tp_size"] == 1


@pytest.mark.parametrize("name", _SINGLE_NODE)
@pytest.mark.parametrize("vendor", ["cuda", "musa"])
def test_other_example_vendors_do_not_receive_hygon_options(monkeypatch, name, vendor):
    kwargs = _load_example(monkeypatch, name, vendor)["_extra_engine_kwargs"]
    assert "cuda_graph_backend_prefill" not in kwargs
    assert "kv_cache_dtype" not in kwargs
    assert kwargs["trust_remote_code"] is True
    if vendor == "musa":
        assert kwargs["page_size"] == 1
    else:
        assert "page_size" not in kwargs


@pytest.mark.parametrize("name", _MULTINODE)
@pytest.mark.parametrize("role", ["master", "worker"])
def test_multinode_hygon_keeps_supported_graph_aliases_and_page_size(
    monkeypatch, name, role
):
    namespace = _load_example(monkeypatch, name)
    path = _ROOT / "examples" / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == f"run_{role}"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cmd"
            for target in node.targets
        )
    )
    args = SimpleNamespace(
        tp=4,
        pp=2,
        port=30000,
        nnodes=2,
        master_addr="192.0.2.1",
        dist_port=20000,
        nccl_port=28765,
    )
    namespace.update(args=args, node_rank=0 if role == "master" else 1)
    # Evaluate only the actual command expression; neither role function runs,
    # so no socket connection, signal handler or subprocess is created.
    cmd = eval(compile(ast.Expression(assignment.value), str(path), "eval"), namespace)
    assert cmd[cmd.index("--page-size") + 1] == "64"
    assert cmd[cmd.index("--tp") + 1] == "4"
    assert cmd[cmd.index("--pp-size") + 1] == "2"
    assert "--disable-radix-cache" in cmd
    assert "--disable-cuda-graph" in cmd
    # 0.5.18 still registers this CLI alias to prefill backend=disabled.
    assert "--disable-piecewise-cuda-graph" in cmd
    assert "--enable-breakable-cuda-graph" not in cmd
    assert "--disable-overlap-schedule" not in cmd
