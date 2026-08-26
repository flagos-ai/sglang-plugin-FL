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

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import gdn
from sglang_fl.patches.triton_kernel import KernelLaunchMetaProxy


class _FakeKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return grid, args, kwargs

        return launch


def _fake_module():
    return SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_FakeKernel()
    )


def test_s5000_defaults_to_eight_warps(monkeypatch):
    module = _fake_module()
    monkeypatch.delenv(gdn._ENV, raising=False)
    monkeypatch.delenv(gdn._LEGACY_ENV, raising=False)
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _name: module)

    gdn.apply_musa_gdn_launch_patch()
    proxy = module.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert isinstance(proxy, KernelLaunchMetaProxy)
    assert proxy.launch_overrides == {"num_warps": 8}

    gdn.apply_musa_gdn_launch_patch()
    assert module.fused_recurrent_gated_delta_rule_packed_decode_kernel is proxy


def test_non_s5000_keeps_sglang_launch_policy(monkeypatch):
    module = _fake_module()
    monkeypatch.delenv(gdn._ENV, raising=False)
    monkeypatch.delenv(gdn._LEGACY_ENV, raising=False)
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S4000")
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _name: module)

    gdn.apply_musa_gdn_launch_patch()

    assert not isinstance(
        module.fused_recurrent_gated_delta_rule_packed_decode_kernel,
        KernelLaunchMetaProxy,
    )


@pytest.mark.parametrize("env_name", [gdn._ENV, gdn._LEGACY_ENV])
def test_explicit_override_has_priority(monkeypatch, env_name):
    module = _fake_module()
    monkeypatch.delenv(gdn._ENV, raising=False)
    monkeypatch.delenv(gdn._LEGACY_ENV, raising=False)
    monkeypatch.setenv(env_name, "4")
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _name: module)

    gdn.apply_musa_gdn_launch_patch()

    proxy = module.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert proxy.launch_overrides == {"num_warps": 4}


def test_plugin_override_takes_priority_over_legacy_override(monkeypatch):
    module = _fake_module()
    monkeypatch.setenv(gdn._ENV, "8")
    monkeypatch.setenv(gdn._LEGACY_ENV, "4")
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _name: module)

    gdn.apply_musa_gdn_launch_patch()

    proxy = module.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert proxy.launch_overrides == {"num_warps": 8}


def test_off_disables_platform_default(monkeypatch):
    module = _fake_module()
    monkeypatch.setenv(gdn._ENV, "off")
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _name: module)

    gdn.apply_musa_gdn_launch_patch()

    assert not isinstance(
        module.fused_recurrent_gated_delta_rule_packed_decode_kernel,
        KernelLaunchMetaProxy,
    )


def test_invalid_override_fails_fast(monkeypatch):
    monkeypatch.setenv(gdn._ENV, "3")
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")

    with pytest.raises(ValueError, match="auto/off/1/2/4/8/16"):
        gdn.apply_musa_gdn_launch_patch()


def test_missing_sglang_kernel_fails_open(monkeypatch, caplog):
    monkeypatch.delenv(gdn._ENV, raising=False)
    monkeypatch.delenv(gdn._LEGACY_ENV, raising=False)
    monkeypatch.setattr(gdn, "_get_musa_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(
        gdn.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    gdn.apply_musa_gdn_launch_patch()

    assert "unavailable in this SGLang revision" in caplog.text
