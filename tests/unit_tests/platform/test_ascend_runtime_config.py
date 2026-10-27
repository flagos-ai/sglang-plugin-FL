# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for early, Ascend-only deterministic reduction defaults."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "sglang_fl/dispatch/backends/vendor/ascend/runtime_config.py"
SPEC = importlib.util.spec_from_file_location("ascend_runtime_contract", MODULE_PATH)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def test_unset_determinism_defaults_true_without_changing_server_args(monkeypatch):
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    args = SimpleNamespace(
        disable_cuda_graph=False,
        disable_overlap_schedule=False,
        dtype="bfloat16",
        max_running_requests=64,
    )
    before = vars(args).copy()
    RUNTIME.apply_ascend_runtime_defaults(args)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    assert vars(args) == before


@pytest.mark.parametrize("explicit", ("true", "false", "strict", ""))
def test_explicit_environment_is_never_overwritten(monkeypatch, explicit):
    monkeypatch.setenv("HCCL_DETERMINISTIC", explicit)
    RUNTIME.apply_ascend_runtime_defaults(SimpleNamespace())
    assert os.environ["HCCL_DETERMINISTIC"] == explicit


def test_repeated_defaults_honor_a_later_explicit_choice(monkeypatch):
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    args = SimpleNamespace()
    RUNTIME.apply_ascend_runtime_defaults(args)
    RUNTIME.apply_ascend_runtime_defaults(args)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    monkeypatch.setenv("HCCL_DETERMINISTIC", "false")
    RUNTIME.apply_ascend_runtime_defaults(args)
    assert os.environ["HCCL_DETERMINISTIC"] == "false"


@pytest.mark.parametrize("vendor", ("ascend", "nvidia", "hygon", "metax"))
def test_actual_platform_callback_only_applies_on_ascend(monkeypatch, vendor):
    # Execute the exact callback without importing SGLang/hardware discovery.
    tree = ast.parse((ROOT / "sglang_fl/platform.py").read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_server_args_defaults"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "platform-callback", "exec"
        ),
        namespace,
    )
    name = "sglang_fl.dispatch.backends.vendor.ascend.runtime_config"
    stub = ModuleType(name)
    calls = []

    def apply(args):
        calls.append(args)
        RUNTIME.apply_ascend_runtime_defaults(args)

    stub.apply_ascend_runtime_defaults = apply
    monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    platform = SimpleNamespace(
        _vendor_name=vendor,
        _device_type="npu" if vendor == "ascend" else "cuda",
        get_default_attention_backend=lambda: "vendor-default",
    )
    args = SimpleNamespace(attention_backend="explicit-backend")
    namespace["apply_server_args_defaults"](platform, args)
    assert calls == ([args] if vendor == "ascend" else [])
    assert os.environ.get("HCCL_DETERMINISTIC") == (
        "true" if vendor == "ascend" else None
    )
    assert args.attention_backend == "explicit-backend"


def test_shell_acceptance_preserves_the_same_explicit_override_policy():
    source = (ROOT / "scripts/ascend/acceptance_common.sh").read_text(encoding="utf-8")
    assert 'export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC-true}"' in source
    assert "printf 'HCCL_DETERMINISTIC=%s\\n'" in source
