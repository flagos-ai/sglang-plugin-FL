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

# Shared fixtures for platform-layer unit tests.

import sys
import types

import pytest


@pytest.fixture
def mock_npu_mamba(monkeypatch):
    """Isolate installer tests from optional SDK imports and global aliases."""
    package = types.ModuleType("sgl_kernel_npu")
    package.__path__ = []
    mamba = types.ModuleType("sgl_kernel_npu.mamba")
    mamba.__path__ = []
    package.mamba = mamba
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, mamba.__name__, mamba)
    return mamba


@pytest.fixture
def cpu_causal_conv():
    """Dense native-call contract using grouped PyTorch convolution on CPU."""
    import torch
    import torch.nn.functional as F

    def native(
        x,
        weight,
        bias=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation="silu",
    ):
        single = x.ndim == 2
        dense = x.unsqueeze(0) if single else x
        width = weight.shape[-1]
        padded = (
            F.pad(dense, (width - 1, 0))
            if initial_states is None
            else torch.cat((initial_states, dense), dim=-1)
        )
        out = F.conv1d(padded, weight.unsqueeze(1), bias, groups=weight.shape[0])
        if activation in ("silu", "swish"):
            out = F.silu(out)
        final = None
        if return_final_states:
            final = padded[..., -(width - 1) :].clone()
            if final_states_out is not None:
                final_states_out.copy_(final)
                final = final_states_out
        return out.squeeze(0) if single else out, final

    return native


@pytest.fixture(params=["cpu", "vendor"])
def native_causal_conv(request, cpu_causal_conv):
    """Always test the CPU contract, and also test the SDK where installed."""
    if request.param == "cpu":
        return cpu_causal_conv
    module = pytest.importorskip("sgl_kernel_npu.mamba.causal_conv1d")
    return module.causal_conv1d_fn_native


@pytest.fixture
def mock_device_detector(monkeypatch):
    """Factory fixture that stubs ``flag_gems.runtime.backend.device.DeviceDetector``.

    Usage:
        mock_device_detector("fakevendor")                  # success path
        mock_device_detector(raise_exc=RuntimeError("..."))  # failure path

    The stub goes into ``sys.modules`` so the production ``from
    flag_gems.runtime.backend.device import DeviceDetector`` hits it without
    needing flag_gems installed.

    Also clears the process-wide ``get_device_info`` lru_cache so each test's
    mock takes effect (otherwise a previous test's cached DeviceInfo would
    win).
    """

    def _make(vendor_name=None, raise_exc=None):
        # Invalidate the DeviceInfo singleton so the new mock is honoured.
        try:
            from sglang_fl.utils import get_device_info

            get_device_info.cache_clear()
        except Exception:
            pass

        fake_mod = types.ModuleType("flag_gems.runtime.backend.device")
        if raise_exc is not None:

            def _detector_fail(*args, **kwargs):
                raise raise_exc

            fake_mod.DeviceDetector = _detector_fail
        else:

            class FakeDetector:
                def __init__(self):
                    self.vendor_name = vendor_name

            fake_mod.DeviceDetector = FakeDetector

        for name in ("flag_gems", "flag_gems.runtime", "flag_gems.runtime.backend"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "flag_gems.runtime.backend.device", fake_mod)

    return _make


@pytest.fixture
def inject_vendor_module(monkeypatch):
    """Factory fixture that injects a fake submodule under
    ``sglang_fl.dispatch.backends.vendor.<vendor>``.

    Usage:
        inject_vendor_module("fakevendor", "register_platform")
        inject_vendor_module("fakevendor", "patch")

    The injected module lives in ``sys.modules`` so ``importlib.import_module``
    returns it directly without touching disk. The fixture cleans up on teardown.
    """

    def _make(vendor_name, submodule):
        pkg_path = f"sglang_fl.dispatch.backends.vendor.{vendor_name}"
        if pkg_path not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg_path, types.ModuleType(pkg_path))

        full = f"{pkg_path}.{submodule}"
        mod = types.ModuleType(full)
        monkeypatch.setitem(sys.modules, full, mod)
        return mod

    return _make
