# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import topk_schedule


def _tensor_shape(*shape):
    return SimpleNamespace(ndim=len(shape), shape=shape)


def test_topk_schedule_is_enabled_by_default_but_can_be_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_MUSA_TOPK_SCHEDULE", raising=False)
    assert topk_schedule._enabled()

    monkeypatch.setenv("SGLANG_MUSA_TOPK_SCHEDULE", "off")
    assert not topk_schedule._enabled()


def test_target_shape_requires_measured_dimensions():
    weights = _tensor_shape(64, 8)
    assert topk_schedule._is_target_shape(weights, _tensor_shape(64, 256), 0, None)
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(65, 256), 0, None)
    assert topk_schedule._is_target_shape(
        _tensor_shape(4095, 8), _tensor_shape(4095, 256), 0, None
    )
    assert topk_schedule._is_target_shape(
        _tensor_shape(15360, 8), _tensor_shape(15360, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        _tensor_shape(16385, 8), _tensor_shape(16385, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(64, 128), 0, None)
    assert not topk_schedule._is_target_shape(
        _tensor_shape(64, 4), _tensor_shape(64, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 1.0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 0, object()
    )


def test_fn_run_signature_ok_accepts_jit_shape_and_rejects_fakes():
    class _Good:
        def run(self, *args, grid, warmup, **kwargs):
            return None

    class _MissingWarmup:
        def run(self, *args, grid, **kwargs):
            return None

    class _NoVarKw:
        def run(self, *args, grid, warmup):
            return None

    assert topk_schedule._fn_run_signature_ok(_Good().run) is True
    assert topk_schedule._fn_run_signature_ok(_MissingWarmup().run) is False
    assert topk_schedule._fn_run_signature_ok(_NoVarKw().run) is False
    assert topk_schedule._fn_run_signature_ok(lambda: None) is False


def test_fn_run_signature_bind_rejects_positional_grid_warmup():
    class _PositionalGridWarmup:
        def run(self, grid, warmup, *args, **kwargs):
            return None

    assert topk_schedule._fn_run_signature_ok(_PositionalGridWarmup().run) is False


def test_fn_run_signature_bind_rejects_extra_required():
    class _ExtraRequired:
        def run(self, *args, grid, warmup, required, **kwargs):
            return None

    assert topk_schedule._fn_run_signature_ok(_ExtraRequired().run) is False


def test_verify_pinned_startup_rejects_unknown_fake_kernel():
    fake_kernel = SimpleNamespace(configs=[object()], fn=SimpleNamespace(run=lambda: None))
    assert topk_schedule._verify_pinned_startup(fake_kernel, object()) is None

    triton = pytest.importorskip("triton")
    real_selected = triton.Config({}, num_warps=1, num_stages=1)
    assert topk_schedule._verify_pinned_startup(fake_kernel, real_selected) is None


def _target_tensors():
    return _tensor_shape(64, 8), object(), _tensor_shape(64, 256)


def test_wrapper_falls_back_when_startup_not_verified():
    configs = [object(), object()]
    kernel = SimpleNamespace(configs=configs, fn=SimpleNamespace(run=lambda: None))

    seen = []

    def original(*args):
        seen.append(args)
        return "result"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, None)
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result == "result"
    assert len(seen) == 1
    assert kernel.configs is configs


def test_wrapper_falls_back_on_runtime_fn_replacement(monkeypatch):
    configs = [object(), object()]

    class _FakeInner:
        def run(self, *args, grid=None, warmup=None, **kwargs):
            raise AssertionError("replaced inner must not run")

    inner = _FakeInner()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {"num_warps": 1, "num_ctas": 1, "num_stages": 1}}
    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)

    seen = []

    def original(*args):
        seen.append(args)
        return "fallback"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, ctx)
    # Runtime replacement after wrapper creation must fail closed.
    kernel.fn = SimpleNamespace(run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result == "fallback"
    assert len(seen) == 1
    assert kernel.configs is configs


@pytest.mark.parametrize("tensor_kind", ["shape_only", "cpu"])
def test_wrapper_falls_back_for_non_musa_tensors(monkeypatch, tensor_kind):
    import torch

    configs = [object(), object()]
    inner = object()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {}}
    seen = []
    guard_calls = []
    device_guard = topk_schedule._is_musa_launch_eligible

    def checked_device_guard(*args):
        guard_calls.append(args)
        return device_guard(*args)

    def original(*args):
        seen.append(args)
        return "fallback"

    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", checked_device_guard)
    monkeypatch.setattr(
        topk_schedule, "_pinned_launch_closed", lambda *a: pytest.fail("must not launch")
    )
    wrapped = topk_schedule._make_topk_wrapper(original, kernel, ctx)
    if tensor_kind == "cpu":
        weights = torch.empty((64, 8))
        ids = torch.empty((64, 8), dtype=torch.int32)
        gating = torch.empty((64, 256))
    else:
        weights, ids, gating = _target_tensors()
    assert wrapped(weights, ids, gating) == "fallback"
    assert len(guard_calls) == 1
    assert len(seen) == 1
    assert all(actual is expected for actual, expected in zip(seen[0][:3], (weights, ids, gating)))
    assert kernel.configs is configs


def test_wrapper_leaves_unmeasured_shape_on_autotuner(monkeypatch):
    configs = [object(), object()]
    inner = object()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {}}
    seen = []

    def original(*args):
        seen.append(kernel.configs)

    monkeypatch.setattr(
        topk_schedule, "_is_musa_launch_eligible", lambda *a: pytest.fail("must short-circuit")
    )
    wrapped = topk_schedule._make_topk_wrapper(original, kernel, ctx)
    wrapped(_tensor_shape(65, 8), object(), _tensor_shape(65, 256))
    assert seen == [configs]
    assert kernel.configs is configs


def test_inner_jit_abi_ok_with_fakes():
    abi_names = list(topk_schedule._EXPECTED_JIT_ARG_NAMES)
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names)) is True
    swapped = list(abi_names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=swapped)) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names[:-1])) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names + ["EXTRA"])) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace()) is False
    assert topk_schedule._inner_jit_abi_ok(object()) is False
    assert topk_schedule._inner_jit_abi_ok(None) is False


def test_wrapper_passthrough_with_mappingproxy_ctx(monkeypatch):
    from types import MappingProxyType

    monkeypatch.setitem(
        sys.modules,
        "triton",
        SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    )

    configs = [object(), object()]
    configs_before = list(configs)
    calls = {}

    class _FakeInner:
        def run(self, *args, grid=None, warmup=None, **kwargs):
            calls["args"] = args
            calls["grid"] = grid
            calls["warmup"] = warmup
            calls["kwargs"] = kwargs
            return None

    inner = _FakeInner()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {
        "inner_fn": inner,
        "pinned_kwargs": MappingProxyType({"num_warps": 1, "num_ctas": 1, "num_stages": 1}),
    }
    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)

    def original(*args):
        raise AssertionError("original must not be called on the pinned path")

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, ctx)
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result is None
    assert calls["grid"] == (64,)
    assert calls["warmup"] is False
    assert calls["kwargs"]["K"] == 8
    assert calls["kwargs"]["num_warps"] == 1
    assert calls["kwargs"]["num_ctas"] == 1
    assert calls["kwargs"]["num_stages"] == 1
    assert kernel.configs is configs
    assert configs == configs_before


def test_wrapper_propagates_launch_failure_without_retry(monkeypatch):
    inner = object()
    kernel = SimpleNamespace(fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {}}
    failure = RuntimeError("launch failure")

    def launch(*args):
        raise failure

    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)
    monkeypatch.setattr(topk_schedule, "_pinned_launch_closed", launch)
    wrapped = topk_schedule._make_topk_wrapper(
        lambda *a: pytest.fail("must not retry original after launch"), kernel, ctx
    )
    with pytest.raises(RuntimeError) as caught:
        wrapped(*_target_tensors())
    assert caught.value is failure


def test_wrapper_falls_back_when_runtime_inner_read_fails(monkeypatch):
    class Kernel:
        @property
        def fn(self):
            raise RuntimeError("inner unavailable")

    ctx = {"inner_fn": object(), "pinned_kwargs": {}}
    calls = []
    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)
    monkeypatch.setattr(
        topk_schedule, "_pinned_launch_closed", lambda *a: pytest.fail("must not launch")
    )
    wrapped = topk_schedule._make_topk_wrapper(
        lambda *args: calls.append(args) or "fallback", Kernel(), ctx
    )
    assert wrapped(*_target_tensors(), renormalize=True) == "fallback"
    assert len(calls) == 1
    assert calls[0][3:] == (True, 0, None)


@pytest.mark.parametrize("verified", [True, False])
def test_apply_verifies_once_and_preserves_both_aliases(monkeypatch, verified):
    selected = object()
    inner = object()
    kernel = SimpleNamespace(fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {}}
    original = lambda *a: "fallback"
    old_alias = lambda *a: "old alias"
    musa_topk = SimpleNamespace(
        topk_softmax=original, topk_softmax_triton_kernel=kernel
    )
    layer_topk = SimpleNamespace(topk_softmax=old_alias)
    for name, module in {
        "triton": SimpleNamespace(Config=lambda *a, **kw: selected),
        "sglang.srt.hardware_backend.musa.kernels": SimpleNamespace(topk=musa_topk),
        "sglang.srt.layers.moe": SimpleNamespace(topk=layer_topk),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(topk_schedule, "_enabled", lambda: True)
    monkeypatch.setattr(topk_schedule, "_device_name", lambda: "MTT S5000")
    verified_calls = []

    def verify(kernel_arg, selected_arg):
        verified_calls.append((kernel_arg, selected_arg))
        return ctx if verified else None

    monkeypatch.setattr(topk_schedule, "_verify_pinned_startup", verify)
    assert topk_schedule.apply_musa_topk_schedule_patch() is verified
    assert verified_calls == [(kernel, selected)]
    if not verified:
        assert musa_topk.topk_softmax is original
        assert layer_topk.topk_softmax is old_alias
        return

    wrapped = musa_topk.topk_softmax
    assert layer_topk.topk_softmax is wrapped
    assert wrapped(*_target_tensors()) == "fallback"
    assert wrapped(*_target_tensors()) == "fallback"
    assert topk_schedule.apply_musa_topk_schedule_patch()
    assert musa_topk.topk_softmax is wrapped
    assert layer_topk.topk_softmax is wrapped
    assert verified_calls == [(kernel, selected)]


def _install_fake_triton(monkeypatch):
    from types import ModuleType

    class Config:
        def __init__(self, all_kwargs=None):
            self._all_kwargs = dict(all_kwargs or topk_schedule._PINNED_OPTION_KWARGS)
            self.pre_hook = None

        def all_kwargs(self):
            return dict(self._all_kwargs)

    class JITFunction:
        arg_names = list(topk_schedule._EXPECTED_JIT_ARG_NAMES)

        def run(self, *args, grid, warmup, **kwargs):
            return None

    class Autotuner:
        def __init__(self):
            self.reset_to_zero = []
            self.restore_value = []
            self.user_defined_pre_hook = False
            self.user_defined_post_hook = False
            self.arg_names = list(topk_schedule._EXPECTED_JIT_ARG_NAMES)
            self.fn = JITFunction()

    root = ModuleType("triton")
    runtime = ModuleType("triton.runtime")
    autotuner = ModuleType("triton.runtime.autotuner")
    jit = ModuleType("triton.runtime.jit")
    autotuner.Config = Config
    autotuner.Autotuner = Autotuner
    jit.JITFunction = JITFunction
    root.runtime = runtime
    runtime.autotuner = autotuner
    runtime.jit = jit
    for name, module in {
        "triton": root,
        "triton.runtime": runtime,
        "triton.runtime.autotuner": autotuner,
        "triton.runtime.jit": jit,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return Config, Autotuner, JITFunction


def test_verify_pinned_startup_rejects_standard_type_subclasses(monkeypatch):
    Config, Autotuner, JITFunction = _install_fake_triton(monkeypatch)

    valid_config = Config()
    valid_kernel = Autotuner()
    # Baseline: with every contract valid, the standard types are accepted.
    assert topk_schedule._verify_pinned_startup(valid_kernel, valid_config) is not None

    class ConfigSubclass(Config):
        pass

    # Each case changes only the object identity; all other checks stay valid.
    assert topk_schedule._verify_pinned_startup(
        valid_kernel, ConfigSubclass()
    ) is None

    class AutotunerSubclass(Autotuner):
        pass

    assert topk_schedule._verify_pinned_startup(
        AutotunerSubclass(), valid_config
    ) is None

    class JITFunctionSubclass(JITFunction):
        pass

    kernel = Autotuner()
    kernel.fn = JITFunctionSubclass()
    assert topk_schedule._verify_pinned_startup(kernel, valid_config) is None
