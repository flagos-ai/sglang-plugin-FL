"""Draft registration must preserve SGLang factory dispatch semantics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "sglang_fl/dispatch/backends/vendor/hygon/patches/speculative_backend.py"


@pytest.fixture
def factory_env(monkeypatch):
    calls = []

    class Factory:
        topk = 1
        speculative_num_steps = 3
        draft_model_runner = object()
        selected = "hcu"

        def _create_backend(self, name, backend_map, message, stamps_children=False):
            calls.append((name, set(backend_map), stamps_children))
            stamp, backend = backend_map[self.selected]()
            backend.prefill_attention_backend_str = stamp
            backend.decode_attention_backend_str = stamp
            if stamps_children:
                for child in backend.attn_backends:
                    child.decode_attention_backend_str = stamp
            return backend

    module = ModuleType("sglang.srt.speculative.draft_utils")
    module.DraftBackendFactory = Factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("_hcu_speculative_patch", PATH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    builds = []

    def create(factory, *, decode):
        builds.append(decode)
        return "hcu", SimpleNamespace(attn_backends=[SimpleNamespace()])

    monkeypatch.setattr(patch, "_create_hcu_draft_backend", create)
    return Factory, patch, calls, builds


@pytest.mark.parametrize("decode", [False, True])
def test_hcu_factory_keeps_upstream_stamping_and_does_not_mutate_map(
    factory_env, decode
):
    factory, patch, calls, builds = factory_env
    patch.register_hcu_speculative_backends()
    source_map = {"triton": lambda: None}
    result = factory()._create_backend(
        "decode_attention_backend", source_map, "error", decode
    )
    assert source_map.keys() == {"triton"}
    assert builds == [decode]
    assert result.prefill_attention_backend_str == "hcu"
    assert result.decode_attention_backend_str == "hcu"
    assert hasattr(result.attn_backends[0], "decode_attention_backend_str") == decode


def test_other_vendor_factory_is_preserved(factory_env):
    factory, patch, calls, builds = factory_env
    patch.register_hcu_speculative_backends()
    runner = factory()
    runner.selected = "other"
    expected = SimpleNamespace()
    actual = runner._create_backend(
        "prefill_attention_backend", {"other": lambda: ("other", expected)}, "error"
    )
    assert actual is expected
    assert actual.decode_attention_backend_str == "other"
    assert builds == []


def test_registration_is_idempotent(factory_env):
    factory, patch, calls, builds = factory_env
    patch.register_hcu_speculative_backends()
    first = factory._create_backend
    patch.register_hcu_speculative_backends()
    assert factory._create_backend is first
    factory()._create_backend("decode_attention_backend", {}, "error", True)
    assert len(calls) == 1
    assert builds == [True]
