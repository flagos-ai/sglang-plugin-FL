# Integration tests for sglang_fl.load_plugin: step order and idempotency.

import pytest


@pytest.fixture
def reset_plugin_loaded(monkeypatch):
    """load_plugin is idempotent via the module-level _plugin_loaded flag.
    Reset it before each test so load_plugin actually executes.
    """
    import sglang_fl

    monkeypatch.setattr(sglang_fl, "_plugin_loaded", False)


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
        monkeypatch.setattr(
            sglang_fl, "_setup_communicator_hooks", lambda: None
        )
        monkeypatch.setattr(
            sglang_fl, "_apply_vendor_patches", _count("vendor_patches")
        )
        monkeypatch.setenv("SGLANG_FL_OOT_ENABLED", "0")

        sglang_fl.load_plugin()
        sglang_fl.load_plugin()
        sglang_fl.load_plugin()

        assert call_count == {"flaggems": 1, "vendor_patches": 1}
