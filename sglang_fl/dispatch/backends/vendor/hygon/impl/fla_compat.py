# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve saved SGLang public FLA functions without re-entering dispatch."""

from sglang_fl.dispatch.fla_patch import get_original


def original_fla(name: str):
    """Return the public function captured before the FL bridge was installed."""
    original = get_original(name)
    if original is None:
        raise RuntimeError(
            f"FLA original '{name}' not available; initialize the FL plugin "
            "before invoking the Hygon FLA backend"
        )
    return original
