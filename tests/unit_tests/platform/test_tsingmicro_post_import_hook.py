# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several TXDA patches may hook the same module: every callback must run."""

import sys
import types

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import _utils


def test_two_callbacks_on_one_module(monkeypatch, tmp_path) -> None:
    name = "txda_hook_probe_mod"
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    calls = []
    _utils._install_post_import_wrap(name, lambda: calls.append("a"))
    _utils._install_post_import_wrap(name, lambda: calls.append("b"))
    mod = __import__(name)
    assert isinstance(mod, types.ModuleType) and mod.VALUE == 1
    assert calls == ["a", "b"]
    _utils._install_post_import_wrap(name, lambda: calls.append("c"))  # already loaded: runs now
    assert calls == ["a", "b", "c"]
