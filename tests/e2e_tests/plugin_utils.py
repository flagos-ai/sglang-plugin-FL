# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared assertions for sglang-plugin-FL end-to-end tests."""


def assert_sglang_fl_plugin_loaded_and_active() -> None:
    """Fail when the general sglang_fl plugin was not loaded and activated."""
    import sglang_fl

    assert sglang_fl.is_plugin_loaded(), (
        "sglang_fl is importable but its general plugin entry point was not invoked. "
        "Check the 'sglang.srt.plugins' entry-point registration."
    )
    assert sglang_fl.is_plugin_active(), (
        "sglang_fl is importable but its general plugin did not finish activation. "
        "Check the 'sglang.srt.plugins' entry-point registration."
    )
