# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CoreX compatibility patches must preserve dtype and import-order contracts."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.iluvatar.patches import (
    clamp_position,
    legacy_gpu_gate,
)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _pdl_probe(out, ENABLE_PDL: tl.constexpr):
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()
        tl.store(out, 7)
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()


def test_clamp_uses_native_implementation_and_can_retry(monkeypatch):
    monkeypatch.setattr(clamp_position, "_applied", False)
    module = SimpleNamespace()
    monkeypatch.setattr(clamp_position.importlib, "import_module", lambda name: module)
    with pytest.raises(AttributeError):
        clamp_position.patch_clamp_position()
    assert not clamp_position._applied

    module._clamp_position_native = lambda lengths: (lengths - 1).clamp(min=0)
    clamp_position.patch_clamp_position()
    assert torch.equal(
        module.clamp_position(torch.tensor([0, 1, 8])), torch.tensor([0, 0, 7])
    )
    assert clamp_position._applied


@pytest.mark.parametrize("supported", [False, True])
def test_bf16_follows_vendor_support_instead_of_sm71(monkeypatch, supported):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 1))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: supported)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_context=lambda: SimpleNamespace(
                override=lambda *a, **kw: calls.append((a, kw))
            )
        ),
    )
    config = SimpleNamespace(dtype=torch.bfloat16)
    legacy_gpu_gate._downgrade_for_iluvatar(server_args=object(), model_config=config)
    assert config.dtype == (torch.bfloat16 if supported else torch.float16)
    assert len(calls) == (0 if supported else 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_explicit_non_bf16_dtype_is_preserved(monkeypatch, dtype):
    def unexpected_query(**kwargs):
        raise AssertionError("non-BF16 dtype must not query CUDA capabilities")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected_query)
    config = SimpleNamespace(dtype=dtype)
    legacy_gpu_gate._downgrade_for_iluvatar(model_config=config)
    assert config.dtype == dtype


def test_older_vendor_bf16_query_without_keyword(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    config = SimpleNamespace(dtype=torch.bfloat16)
    legacy_gpu_gate._downgrade_for_iluvatar(model_config=config)
    assert config.dtype == torch.bfloat16


def test_missing_bf16_query_requires_explicit_dtype(monkeypatch):
    monkeypatch.delattr(torch.cuda, "is_bf16_supported")
    with pytest.raises(RuntimeError, match="select --dtype float16"):
        legacy_gpu_gate._downgrade_for_iluvatar(
            model_config=SimpleNamespace(dtype=torch.bfloat16)
        )


def test_legacy_gate_refreshes_copied_helper_and_can_retry(monkeypatch):
    monkeypatch.setattr(legacy_gpu_gate, "_patched", False)
    module = SimpleNamespace()
    monkeypatch.setattr(legacy_gpu_gate.importlib, "import_module", lambda name: module)
    with pytest.raises(AttributeError):
        legacy_gpu_gate.patch_legacy_gpu_gate()
    assert not legacy_gpu_gate._patched

    def original(**kw):
        return None

    module.maybe_downgrade_dtype_for_legacy_gpu = original
    runner = SimpleNamespace(maybe_downgrade_dtype_for_legacy_gpu=original)
    monkeypatch.setitem(sys.modules, legacy_gpu_gate._MODEL_RUNNER_MODULE, runner)
    legacy_gpu_gate.patch_legacy_gpu_gate()
    assert (
        module.maybe_downgrade_dtype_for_legacy_gpu
        is legacy_gpu_gate._downgrade_for_iluvatar
    )
    assert (
        runner.maybe_downgrade_dtype_for_legacy_gpu
        is legacy_gpu_gate._downgrade_for_iluvatar
    )


def _load_pdl_with_fake_triton(monkeypatch, cuda):
    triton, language = ModuleType("triton"), ModuleType("triton.language")
    triton.jit = lambda fn: fn
    language.extra = SimpleNamespace(cuda=cuda)
    language.static_assert = lambda condition, message: (
        None if condition else (_ for _ in ()).throw(RuntimeError(message))
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(legacy_gpu_gate.__file__).with_name("triton_pdl.py")
    spec = importlib.util.spec_from_file_location("_iluvatar_test_pdl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdl_is_false_missing_names_fail_and_real_intrinsic_is_preserved(monkeypatch):
    real_wait = object()
    cuda = SimpleNamespace(gdc_wait=real_wait)
    pdl = _load_pdl_with_fake_triton(monkeypatch, cuda)

    def original():
        return True

    utils = SimpleNamespace(is_arch_support_pdl=original)
    consumer = SimpleNamespace(is_arch_support_pdl=original)
    monkeypatch.setitem(sys.modules, "sgl_kernel.utils", utils)
    monkeypatch.setitem(sys.modules, "sglang._iluvatar_test_consumer", consumer)

    pdl.patch_triton_pdl_intrinsics()
    pdl.patch_triton_pdl_intrinsics()
    assert utils.is_arch_support_pdl() is False
    assert consumer.is_arch_support_pdl() is False
    assert cuda.gdc_wait is real_wait
    with pytest.raises(RuntimeError, match="not supported on Iluvatar CoreX"):
        cuda.gdc_launch_dependents()


def test_pdl_missing_package_does_not_mark_patch_complete(monkeypatch):
    pdl = _load_pdl_with_fake_triton(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(
        pdl.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing kernel shim")),
    )
    with pytest.raises(ImportError, match="missing kernel shim"):
        pdl.patch_triton_pdl_intrinsics()
    assert not pdl._applied


def test_overlay_runs_pdl_before_sglang_imports_and_retries(monkeypatch):
    calls = []
    fail = [True]

    def clamp():
        calls.append("clamp")
        if fail[0]:
            raise RuntimeError("transient import failure")

    prefix = "sglang_fl.dispatch.backends.vendor.iluvatar.patches"
    monkeypatch.setitem(
        sys.modules,
        prefix + ".triton_pdl",
        SimpleNamespace(patch_triton_pdl_intrinsics=lambda: calls.append("pdl")),
    )
    monkeypatch.setattr(clamp_position, "patch_clamp_position", clamp)
    monkeypatch.setattr(
        legacy_gpu_gate,
        "patch_legacy_gpu_gate",
        lambda: calls.append("legacy"),
    )
    path = Path(legacy_gpu_gate.__file__).parents[1] / "patch.py"
    spec = importlib.util.spec_from_file_location(
        "sglang_fl.dispatch.backends.vendor.iluvatar._test_overlay",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="transient import failure"):
        spec.loader.exec_module(module)
    assert not module._patches_applied
    fail[0] = False
    module.apply_iluvatar_patches()
    module.apply_iluvatar_patches()
    assert calls == ["pdl", "clamp", "pdl", "clamp", "legacy"]


@pytest.mark.gpu
def test_corex_dead_pdl_branch_compiles_and_live_pdl_is_rejected(monkeypatch):
    if triton is None or not hasattr(torch, "corex") or not torch.cuda.is_available():
        pytest.skip("CoreX hardware and vendor Triton are required")
    pdl = importlib.import_module(
        "sglang_fl.dispatch.backends.vendor.iluvatar.patches.triton_pdl"
    )
    monkeypatch.setattr(pdl, "_applied", False)
    monkeypatch.setattr(tl.extra, "cuda", SimpleNamespace(), raising=False)
    utils = importlib.import_module("sgl_kernel.utils")
    monkeypatch.setattr(utils, "is_arch_support_pdl", lambda: True)
    pdl.patch_triton_pdl_intrinsics()
    assert utils.is_arch_support_pdl() is False
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    _pdl_probe[(1,)](output, False)
    assert output.item() == 7
    with pytest.raises(triton.CompilationError) as error:
        _pdl_probe[(1,)](output, True)
    causes, cause = [], error.value
    while cause is not None:
        causes.append(str(cause))
        cause = cause.__cause__
    assert "CUDA PDL is not supported on Iluvatar CoreX" in "\n".join(causes)
