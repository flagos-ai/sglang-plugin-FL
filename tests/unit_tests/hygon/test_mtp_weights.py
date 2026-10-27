# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of HCU MTP load coverage and deferred import interception."""

import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/backends/"
    "vendor/hygon/patches/mtp_weights.py"
)
TARGET = "sglang.srt.models.qwen3_5_mtp"
ANCHORS = {
    "fc.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "model.norm.weight",
}
DENSE = ANCHORS | {
    "model.layers.0.qkv_proj.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
}
SHARED = {"model.embed_tokens.weight", "lm_head.weight"}
DENSE_KEYS = {
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
}


@pytest.fixture
def guard(monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_hcu_mtp_guard", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(__hcu_version__="3.6")
    )
    monkeypatch.delitem(sys.modules, TARGET, raising=False)
    yield module
    if module._finder in sys.meta_path:
        sys.meta_path.remove(module._finder)
    sys.modules.pop(TARGET, None)


def model_class(params=DENSE, loaded=None, num_experts=None, failure=None):
    class Qwen3_5ForCausalLMMTP:
        def __init__(self):
            self.config = types.SimpleNamespace(num_experts=num_experts)
            self.result = set(params if loaded is None else loaded)
            self.seen = []

        def named_parameters(self):
            return ((name, object()) for name in sorted(set(params) | SHARED))

        def load_weights(self, weights, is_mtp=False):
            self.is_mtp = is_mtp
            for name, tensor in weights:
                self.seen.append((name, tensor))
            if failure is not None:
                raise failure
            return self.result

    Qwen3_5ForCausalLMMTP.__module__ = TARGET
    return Qwen3_5ForCausalLMMTP


def checkpoint(keys=DENSE_KEYS):
    # Opaque values prove that the guard needs no tensor data or device operations.
    return iter((name, object()) for name in sorted(keys | {"model.layers.0.weight"}))


def test_dense_coverage_uses_real_return_and_preserves_stream_api(guard, caplog):
    cls = model_class()
    guard._guard_class(cls)
    instance = cls()
    with caplog.at_level("INFO"):
        result = instance.load_weights(checkpoint(), is_mtp=True)
    assert result is instance.result and instance.is_mtp is True
    report = instance._hcu_mtp_weight_coverage
    assert set(report["required_params"]) == DENSE
    assert set(report["shared_params"]) == SHARED
    assert report["missing_params"] == report["missing_components"] == []
    assert report["checkpoint_tensors_consumed"] == len(DENSE_KEYS) + 1
    assert len(instance.seen) == len(DENSE_KEYS) + 1
    assert "Qwen3_5ForCausalLMMTP required=8 loaded=8 shared=2" in caplog.text


@pytest.mark.parametrize(
    "missing", ["fc.weight", "model.norm.weight", "model.layers.0.qkv_proj.weight"]
)
def test_returned_parameter_coverage_must_include_each_head_parameter(guard, missing):
    cls = model_class(loaded=DENSE - {missing})
    guard._guard_class(cls)
    with pytest.raises(RuntimeError, match="missing parameters"):
        cls().load_weights(checkpoint())


@pytest.mark.parametrize(
    "key",
    [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
    ],
)
def test_fused_parameter_return_cannot_hide_a_missing_source_component(guard, key):
    cls = model_class()
    guard._guard_class(cls)
    with pytest.raises(RuntimeError, match="missing checkpoint components"):
        cls().load_weights(checkpoint(DENSE_KEYS - {key}))


def test_target_only_checkpoint_does_not_initialize_mtp(guard):
    cls = model_class(loaded=set())
    guard._guard_class(cls)
    with pytest.raises(RuntimeError, match="HCU MTP weights incomplete"):
        cls().load_weights(iter([("model.layers.0.weight", object())]))


@pytest.mark.parametrize("fused", [True, False])
def test_moe_fused_or_all_numbered_expert_components_are_required(guard, fused):
    experts = {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }
    cls = model_class(params=DENSE | experts, num_experts=2)
    guard._guard_class(cls)
    if fused:
        keys = {
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        }
    else:
        keys = {
            f"mtp.layers.0.mlp.experts.{i}.{part}_proj.weight"
            for i in range(2)
            for part in ("gate", "up", "down")
        }
    cls().load_weights(checkpoint(DENSE_KEYS | keys))
    with pytest.raises(RuntimeError, match="missing checkpoint components"):
        cls().load_weights(checkpoint(DENSE_KEYS | (keys - {sorted(keys)[0]})))


def test_full_head_structure_and_return_type_fail_closed(guard):
    cls = model_class(params={"model.layers.0.input_layernorm.weight"})
    guard._guard_class(cls)
    with pytest.raises(RuntimeError, match="no complete MTP branch"):
        cls().load_weights(checkpoint())
    cls = model_class()
    guard._guard_class(cls)
    instance = cls()
    instance.result = None
    with pytest.raises(RuntimeError, match="must return loaded parameter names"):
        instance.load_weights(checkpoint())


def test_upstream_error_propagates_without_verified_log(guard, caplog):
    error = ValueError("real weight loader failure")
    cls = model_class(failure=error)
    guard._guard_class(cls)
    with caplog.at_level("INFO"), pytest.raises(ValueError) as exc:
        cls().load_weights(checkpoint())
    assert exc.value is error
    assert "weights verified" not in caplog.text


def test_guard_is_idempotent_and_never_changes_other_vendor_loading(guard, monkeypatch):
    cls = model_class()
    guard._guard_class(cls)
    first = cls.load_weights
    guard._guard_class(cls)
    assert cls.load_weights is first
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    instance = cls()
    assert instance.load_weights(iter([])) is instance.result
    assert not hasattr(instance, "_hcu_mtp_weight_coverage")
    assert guard.register_hcu_mtp_weight_guard() is False
    assert guard._finder is None


def test_foreign_model_class_is_rejected(guard):
    cls = model_class()
    cls.__module__ = "another_vendor.models.qwen"
    with pytest.raises(RuntimeError, match="unexpected model class"):
        guard._guard_class(cls)


def test_already_imported_mtp_is_patched_without_registry_import(guard, monkeypatch):
    cls = model_class()
    monkeypatch.setitem(
        sys.modules, TARGET, types.SimpleNamespace(Qwen3_5ForCausalLMMTP=cls)
    )
    registry_before = sys.modules.get("sglang.srt.models.registry")
    assert guard.register_hcu_mtp_weight_guard() is True
    assert getattr(cls.load_weights, guard._MARKER, False)
    assert guard._finder is None
    assert sys.modules.get("sglang.srt.models.registry") is registry_before


def test_setup_defers_import_then_wraps_only_natural_mtp_import(guard, monkeypatch):
    calls = []
    cls = model_class()

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            calls.append(module.__name__)
            module.Qwen3_5ForCausalLMMTP = cls

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == TARGET:
                return importlib.util.spec_from_loader(fullname, Loader())
            return None

    for name in ("sglang", "sglang.srt", "sglang.srt.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    finder = Finder()
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))
    registry_before = sys.modules.get("sglang.srt.models.registry")
    assert guard.register_hcu_mtp_weight_guard() is True
    assert guard.register_hcu_mtp_weight_guard() is True
    assert calls == [] and TARGET not in sys.modules
    assert sys.modules.get("sglang.srt.models.registry") is registry_before
    assert sum(item is guard._finder for item in sys.meta_path) == 1
    assert guard._finder.find_spec("unrelated.cuda.model") is None
    module = importlib.import_module(TARGET)
    assert calls == [TARGET] and module.Qwen3_5ForCausalLMMTP is cls
    assert guard._finder not in sys.meta_path
    cls().load_weights(checkpoint())


def _natural_model_finder(
    monkeypatch, cls, *, entered=None, release=None, callback=None
):
    """An actual importlib loader; events control class-definition timing."""
    calls = []

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            calls.append(module.__name__)
            if entered is not None:
                entered.set()
            if release is not None and not release.wait(timeout=5):
                raise RuntimeError("test loader release timed out")
            if callback is not None:
                callback()
            module.Qwen3_5ForCausalLMMTP = cls

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == TARGET:
                return importlib.util.spec_from_loader(fullname, Loader())
            return None

    for name in ("sglang", "sglang.srt", "sglang.srt.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    finder = Finder()
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))
    return calls


def test_concurrent_first_registration_installs_one_finder_and_imports_naturally(
    guard, monkeypatch
):
    import threading

    cls = model_class()
    calls = _natural_model_finder(monkeypatch, cls)
    original_finder = guard._MTPImportFinder
    first_created = threading.Event()
    second_created = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    constructed = 0

    class BlockingFinder(original_finder):
        def __init__(self):
            nonlocal constructed
            with counter_lock:
                constructed += 1
                index = constructed
            if index == 1:
                first_created.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test registration release timed out")
            else:
                second_created.set()

    monkeypatch.setattr(guard, "_MTPImportFinder", BlockingFinder)
    errors, results = [], []

    def register(second=False):
        try:
            if second:
                second_started.set()
            results.append(guard.register_hcu_mtp_weight_guard())
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=register, daemon=True),
        threading.Thread(target=register, args=(True,), daemon=True),
    ]
    try:
        threads[0].start()
        assert first_created.wait(timeout=5)
        threads[1].start()
        assert second_started.wait(timeout=5)
        # Give a competing registration a chance to enter the deliberately
        # blocked constructor. A correct implementation serializes before it.
        second_created.wait(timeout=0.1)
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == [] and results == [True, True]
    assert constructed == 1
    assert sum(isinstance(item, original_finder) for item in sys.meta_path) == 1
    assert calls == [] and TARGET not in sys.modules
    imported = importlib.import_module(TARGET)
    assert calls == [TARGET]
    assert getattr(imported.Qwen3_5ForCausalLMMTP.load_weights, guard._MARKER, False)
    assert guard._finder not in sys.meta_path
    cls().load_weights(checkpoint())


@pytest.mark.parametrize("pre_registered", [False, True])
def test_registration_waits_for_natural_import_without_deadlock(
    guard, monkeypatch, pre_registered
):
    import threading

    cls = model_class()
    entered = threading.Event()
    release = threading.Event()
    registration_started = threading.Event()
    registration_done = threading.Event()
    calls = _natural_model_finder(monkeypatch, cls, entered=entered, release=release)
    if pre_registered:
        assert guard.register_hcu_mtp_weight_guard()
    errors, results = [], []

    def natural_import():
        try:
            results.append(importlib.import_module(TARGET))
        except BaseException as error:
            errors.append(error)

    def register():
        try:
            registration_started.set()
            results.append(guard.register_hcu_mtp_weight_guard())
        except BaseException as error:
            errors.append(error)
        finally:
            registration_done.set()

    importer = threading.Thread(target=natural_import, daemon=True)
    registrar = threading.Thread(target=register, daemon=True)
    try:
        importer.start()
        assert entered.wait(timeout=5)
        partial = sys.modules[TARGET]
        assert partial.__spec__._initializing
        assert not hasattr(partial, "Qwen3_5ForCausalLMMTP")
        registrar.start()
        assert registration_started.wait(timeout=5)
        assert not registration_done.wait(timeout=0.1)
    finally:
        release.set()
        importer.join(timeout=5)
        if registrar.ident is not None:
            registrar.join(timeout=5)
    assert not importer.is_alive() and not registrar.is_alive()
    assert errors == []
    assert True in results
    assert calls == [TARGET]
    imported = sys.modules[TARGET]
    assert not imported.__spec__._initializing
    assert imported.Qwen3_5ForCausalLMMTP is cls
    assert getattr(cls.load_weights, guard._MARKER, False)
    assert guard._finder not in sys.meta_path
    first = cls.load_weights
    assert guard.register_hcu_mtp_weight_guard()
    assert cls.load_weights is first
    cls().load_weights(checkpoint())


def test_registration_reentry_from_guarded_loader_defers_until_class_exists(
    guard, monkeypatch
):
    cls = model_class()
    outcomes = []
    calls = _natural_model_finder(
        monkeypatch,
        cls,
        callback=lambda: outcomes.append(guard.register_hcu_mtp_weight_guard()),
    )
    assert guard.register_hcu_mtp_weight_guard()
    imported = importlib.import_module(TARGET)
    assert outcomes == [True] and calls == [TARGET]
    assert imported.Qwen3_5ForCausalLMMTP is cls
    assert getattr(cls.load_weights, guard._MARKER, False)
    cls().load_weights(checkpoint())
