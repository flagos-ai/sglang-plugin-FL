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

# Integration tests for sglang_fl.load_plugin: step order and idempotency.

from types import SimpleNamespace

import pytest


@pytest.fixture
def reset_plugin_loaded(monkeypatch):
    """load_plugin is idempotent via the module-level _plugin_loaded flag.
    Reset it before each test so load_plugin actually executes.
    """
    import sglang_fl

    monkeypatch.setattr(sglang_fl, "_plugin_loaded", False)
    monkeypatch.setattr(sglang_fl, "_plugin_active", False)
    monkeypatch.delenv("SGLANG_FL_MODE", raising=False)


class TestLoadPluginStepOrder:
    def test_vendor_patches_run_after_all_sglang_fl_layers(
        self, monkeypatch, reset_plugin_loaded
    ):
        """Core contract: vendor patches MUST execute after every sglang_fl
        baseline layer (FlagGems → dispatch → communicator).

        This is what makes ``vendor/<name>/patch.py`` last-writer-wins — a
        vendor patch can override any ATen op FlagGems replaced, any fused-op
        dispatch table entry, or the communicator hook. If this ordering is
        ever reversed, vendors lose that ability silently.
        """
        import sglang_fl

        call_order = []

        def _record(name):
            return lambda *a, **kw: call_order.append(name)

        monkeypatch.setattr(sglang_fl, "_setup_flaggems", _record("flaggems"))
        monkeypatch.setattr(sglang_fl, "_init_dispatch", _record("dispatch"))
        monkeypatch.setattr(
            sglang_fl, "_setup_communicator_hooks", _record("communicator")
        )
        monkeypatch.setattr(
            sglang_fl, "_apply_vendor_patches", _record("vendor_patches")
        )

        # Skip the OOT block — its inline calls (HookRegistry, fla_patch,
        # rotary_patch) would need extra mocking but don't affect ordering.
        monkeypatch.setenv("SGLANG_FL_OOT_ENABLED", "0")

        sglang_fl.load_plugin()

        assert call_order == [
            "flaggems",
            "dispatch",
            "communicator",
            "vendor_patches",
        ], f"vendor_patches must be last; got {call_order}"


class TestLoadPluginIdempotency:
    def test_repeated_calls_execute_helpers_only_once(
        self, monkeypatch, reset_plugin_loaded
    ):
        """load_plugin sets _plugin_loaded=True after first run and
        short-circuits on subsequent calls — protects against double-init
        when multiple entry-point loaders or test code call it.
        """
        import sglang_fl

        call_count = {"flaggems": 0, "vendor_patches": 0}

        def _count(key):
            def _inc(*a, **kw):
                call_count[key] += 1

            return _inc

        monkeypatch.setattr(sglang_fl, "_setup_flaggems", _count("flaggems"))
        monkeypatch.setattr(sglang_fl, "_init_dispatch", lambda *a, **kw: None)
        monkeypatch.setattr(sglang_fl, "_setup_communicator_hooks", lambda: None)
        monkeypatch.setattr(
            sglang_fl, "_apply_vendor_patches", _count("vendor_patches")
        )
        monkeypatch.setenv("SGLANG_FL_OOT_ENABLED", "0")

        sglang_fl.load_plugin()
        sglang_fl.load_plugin()
        sglang_fl.load_plugin()

        assert call_count == {"flaggems": 1, "vendor_patches": 1}


class TestPlatformProfileMode:
    def test_framework_fallback_uses_active_device_method(self, monkeypatch):
        import sglang.srt.platforms
        import sglang_fl

        monkeypatch.setattr(
            sglang.srt.platforms,
            "current_platform",
            SimpleNamespace(device_type="cuda"),
        )

        class Op:
            def forward_cuda(self):
                return "sglang_cuda"

            def forward_native(self):
                return "sglang_native"

        assert sglang_fl._framework_forward(Op())() == "sglang_cuda"

    def test_framework_fallback_uses_native_without_device_method(self, monkeypatch):
        import sglang.srt.platforms
        import sglang_fl

        monkeypatch.setattr(
            sglang.srt.platforms,
            "current_platform",
            SimpleNamespace(device_type="custom_accelerator"),
        )

        class Op:
            def forward_native(self):
                return "sglang_native"

        assert sglang_fl._framework_forward(Op())() == "sglang_native"

    def test_disables_flaggems_but_keeps_platform_layers(
        self, monkeypatch, reset_plugin_loaded
    ):
        import sglang_fl
        import sglang_fl.profiling_hooks

        calls = []
        captured_config = {}

        monkeypatch.setenv("SGLANG_FL_MODE", "platform_profile")
        monkeypatch.setenv("SGLANG_FL_OOT_ENABLED", "0")
        monkeypatch.setattr(
            sglang_fl.profiling_hooks,
            "setup_operator_profile_hooks",
            lambda: calls.append("profile"),
        )
        monkeypatch.setattr(
            sglang_fl,
            "_setup_flaggems",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("FlagGems must remain disabled")
            ),
        )

        def _dispatch(config):
            captured_config.update(config)
            calls.append("dispatch")

        monkeypatch.setattr(sglang_fl, "_init_dispatch", _dispatch)
        monkeypatch.setattr(
            sglang_fl,
            "_setup_communicator_hooks",
            lambda: calls.append("communicator"),
        )
        monkeypatch.setattr(
            sglang_fl,
            "_apply_vendor_patches",
            lambda: calls.append("vendor_patches"),
        )

        sglang_fl.load_plugin()

        assert calls == ["profile", "dispatch", "communicator", "vendor_patches"]
        assert captured_config["prefer"] == "vendor"
        assert captured_config["strict"] is True
        assert captured_config["op_backends"] == {}
        assert captured_config["deny_vendors"] == set()
        assert captured_config["allow_vendors"] is None
        assert captured_config["oot_blacklist"] == []
        assert captured_config["oot_whitelist"] == []
