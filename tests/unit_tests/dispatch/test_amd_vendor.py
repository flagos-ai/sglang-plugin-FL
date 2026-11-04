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

# Tests for the AMD ROCm vendor backend and its fused-MoE patch.
# Only the hardware layer (torch.version.hip, torch.cuda, current_platform) and
# sglang / FlagGems modules are mocked.

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

AMD_OPS = {
    "fused_moe",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
}


def _platform_modules(vendor_name):
    return {
        "sglang.srt.platforms": MagicMock(
            current_platform=SimpleNamespace(_vendor_name=vendor_name)
        )
    }


@pytest.fixture
def amd_backend_cls():
    from sglang_fl.dispatch.backends.vendor.amd.amd import AmdBackend

    AmdBackend._available = None
    yield AmdBackend
    AmdBackend._available = None


@pytest.mark.parametrize(
    "hip, vendor, expected",
    [("7.2", "amd", True), ("7.2", "hygon", False), (None, "amd", False)],
)
def test_amd_backend_availability(amd_backend_cls, hip, vendor, expected):
    with patch.object(torch.version, "hip", hip), patch(
        "torch.cuda.is_available", return_value=True
    ), patch.dict("sys.modules", _platform_modules(vendor)):
        assert amd_backend_cls().is_available() is expected


def test_register_builtins_wires_availability(amd_backend_cls):
    from sglang_fl.dispatch.backends.vendor.amd.register_ops import register_builtins

    registry = MagicMock()
    register_builtins(registry)
    impls = registry.register_many.call_args.args[0]

    assert {impl.op_name for impl in impls} == AMD_OPS
    assert all(impl.impl_id == "vendor.amd" and impl.vendor == "amd" for impl in impls)
    for available in (True, False):
        amd_backend_cls._available = available
        assert all(impl.is_available() is available for impl in impls)


# ---------------------------------------------------------------- patch.py


class _StubLike:
    """Mimics an import-only stub: every attribute lookup returns a truthy object."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _StubLike()

    def __call__(self, *args, **kwargs):
        raise RuntimeError("stub called")


@pytest.fixture
def amd_patch(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.amd import patch as amd_patch

    monkeypatch.setattr(amd_patch, "_use_flaggems_cache", {})
    return amd_patch


@pytest.fixture
def fake_triton_utils(monkeypatch):
    pkg = "sglang.srt.layers.moe.moe_runner.triton_utils"
    align = types.ModuleType(f"{pkg}.moe_align_block_size")
    align.sgl_moe_align_block_size = _StubLike()
    fused = types.ModuleType(f"{pkg}.fused_moe")
    fused.silu_and_mul = MagicMock(name="sgl_silu_and_mul")
    fused.gelu_and_mul = MagicMock(name="sgl_gelu_and_mul")
    monkeypatch.setitem(sys.modules, align.__name__, align)
    monkeypatch.setitem(sys.modules, fused.__name__, fused)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    return align, fused


def test_patch_wraps_stub_like_originals_once(amd_patch, fake_triton_utils):
    align, fused = fake_triton_utils
    originals = (align.sgl_moe_align_block_size, fused.silu_and_mul, fused.gelu_and_mul)

    amd_patch.patch()
    wrapped = (align.sgl_moe_align_block_size, fused.silu_and_mul, fused.gelu_and_mul)
    amd_patch.patch()

    assert all(w is not o for w, o in zip(wrapped, originals))
    assert all(getattr(w, "_flagos_amd_patched", False) is True for w in wrapped)
    assert (
        align.sgl_moe_align_block_size,
        fused.silu_and_mul,
        fused.gelu_and_mul,
    ) == wrapped


def test_patch_is_noop_without_hip(amd_patch, fake_triton_utils):
    _align, fused = fake_triton_utils
    original = fused.silu_and_mul
    with patch.object(torch.version, "hip", None):
        amd_patch.patch()
    assert fused.silu_and_mul is original


@pytest.mark.parametrize(
    "arch, expected",
    [("gfx1201", True), ("gfx942:sramecc+:xnack-", False), ("gfx950", False)],
)
def test_use_flaggems_by_arch(amd_patch, arch, expected):
    props = SimpleNamespace(gcnArchName=arch)
    with patch("torch.cuda.get_device_properties", return_value=props):
        assert amd_patch._use_flaggems(torch.device("cuda", 0)) is expected


def test_moe_align_routes_by_arch(amd_patch, monkeypatch):
    gems = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "flag_gems.fused.moe_align_block_size",
        SimpleNamespace(moe_align_block_size_triton=gems),
    )
    original = MagicMock()
    wrapped = amd_patch._wrap_moe_align_block_size(original)
    topk_ids = SimpleNamespace(device=torch.device("cuda", 0))
    args = (topk_ids, 257, 16, "sorted", "experts", "ntp")

    monkeypatch.setattr(amd_patch, "_use_flaggems", lambda device: True)
    wrapped(*args, "cumsum", True)
    gems.assert_called_once_with(*args)
    original.assert_not_called()

    monkeypatch.setattr(amd_patch, "_use_flaggems", lambda device: False)
    wrapped(*args, "cumsum", True)
    original.assert_called_once_with(*args, "cumsum", True)


def test_silu_and_mul_splits_gate_and_up(amd_patch, monkeypatch):
    gems_out = MagicMock(side_effect=lambda a, b, out: out.copy_(a * b))
    monkeypatch.setitem(
        sys.modules,
        "flag_gems.fused.silu_and_mul",
        SimpleNamespace(silu_and_mul_out=gems_out),
    )
    monkeypatch.setattr(amd_patch, "_use_flaggems", lambda device: True)
    original = MagicMock()
    wrapped = amd_patch._wrap_silu_and_mul(original)

    x = torch.arange(8.0).view(2, 4)
    out = torch.empty(2, 2)
    wrapped(x, out)
    assert torch.equal(out, x[:, :2] * x[:, 2:])

    wrapped(x, out, expert_ids=None)
    original.assert_called_once()
