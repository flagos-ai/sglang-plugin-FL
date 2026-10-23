# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Selection tests for platform-profile FLA patching."""

from sglang_fl.dispatch.fla_patch import _select_fla_implementations


def test_vendor_only_fla_selection_retains_unavailable_sglang_functions():
    originals = {"chunk": object(), "decode": object()}
    bridges = {"chunk": object(), "decode": object()}

    selected = _select_fla_implementations(
        originals,
        bridges,
        vendor_only=True,
        is_vendor_available=lambda name: name == "chunk",
    )

    assert selected["chunk"] is bridges["chunk"]
    assert selected["decode"] is originals["decode"]


def test_adapt_mode_fla_selection_keeps_all_dispatch_bridges():
    originals = {"chunk": object(), "decode": object()}
    bridges = {"chunk": object(), "decode": object()}

    selected = _select_fla_implementations(
        originals,
        bridges,
        vendor_only=False,
    )

    assert selected == bridges
