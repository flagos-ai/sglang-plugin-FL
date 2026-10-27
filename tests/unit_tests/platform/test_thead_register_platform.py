"""init_backend imports vendor/thead/register_platform, which must apply the thead patches.

Covers entry points such as sglang.bench_one_batch that never call load_plugins().
"""

import importlib
import sys
from types import ModuleType


def test_register_platform_imports_patch_module(monkeypatch):
    pkg = "sglang_fl.dispatch.backends.vendor.thead"
    fake_patch = ModuleType(f"{pkg}.patch")
    monkeypatch.setitem(sys.modules, f"{pkg}.patch", fake_patch)
    monkeypatch.delitem(sys.modules, f"{pkg}.register_platform", raising=False)

    module = importlib.import_module(f"{pkg}.register_platform")

    # The import must go through patch.py (whose import applies the patches).
    assert module.patch is fake_patch
    monkeypatch.delitem(sys.modules, f"{pkg}.register_platform", raising=False)
