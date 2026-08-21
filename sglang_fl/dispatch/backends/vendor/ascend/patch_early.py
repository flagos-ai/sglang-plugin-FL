"""Ascend load-time (T1) patches — applied on ``sglang_fl`` import.

This module is auto-discovered by ``_apply_vendor_early_patches()`` in
``sglang_fl/__init__.py``: when the runtime vendor is ``ascend``, this file
is imported before any sglang module load, so its module-level side effects
can install import-time hooks.

Currently installs:
  - ``SglKernelNpuStubFinder``: an ``sys.meta_path`` finder that redirects
    ``sgl_kernel_npu.*`` imports in srt_empty mode (no sgl_kernel_npu
    package installed). Two per-submodule modes:

    1) Vendored redirect (in ``_VENDORED_ROUTES``): route
       ``sgl_kernel_npu.X.Y`` to a plugin-owned implementation module. Used
       when an sgl_kernel_npu symbol is actually called at runtime in empty
       mode and needs a real implementation.

    2) None stub (everything else): return a module whose attributes are
       all None. Enables sglang core modules to import at load time without
       an ImportError; any actual call will NoneType-fail — which the
       plugin dispatch fallback catches and re-routes to flag_gems /
       reference.

Companion runtime (T2) patches live in ``patch.py`` (siblings in ``patches/``).
"""

from __future__ import annotations

import importlib
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

# Fully-qualified sgl_kernel_npu module name -> plugin-owned implementation module.
# Add an entry here whenever a Class 2 runtime failure surfaces.
_VENDORED_ROUTES: dict[str, str] = {
    "sgl_kernel_npu.mem_cache.allocator": "sglang_fl.dispatch.backends.vendor.ascend.impl.mem_cache.allocator",
    "sgl_kernel_npu.mamba.causal_conv1d": "sglang_fl.dispatch.backends.vendor.ascend.impl.mamba.causal_conv1d",
    "sgl_kernel_npu.fla.fused_gdn_gating": "sglang_fl.dispatch.backends.vendor.ascend.impl.fla.fused_gdn_gating",
    "sgl_kernel_npu.fla.layernorm_gated": "sglang_fl.dispatch.backends.vendor.ascend.impl.fla.layernorm_gated",
    "sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent": "sglang_fl.dispatch.backends.vendor.ascend.impl.fla.fused_sigmoid_gating_recurrent",
}


class _StubLoader(Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__spec__ = spec
        mod.__srt_empty_stub__ = True
        return mod

    def exec_module(self, module):
        module.__getattr__ = lambda name: None  # type: ignore[assignment]


class _VendoredRedirectLoader(Loader):
    """Loader that transparently redirects attribute lookups to a vendored module."""

    def __init__(self, target_module: str):
        self._target = target_module

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__spec__ = spec
        mod.__srt_empty_vendored__ = self._target
        return mod

    def exec_module(self, module):
        target = importlib.import_module(self._target)
        # Any attribute lookup on this stub module goes to the vendored module.
        # Missing attributes raise AttributeError from the target, which is the
        # correct signal ("vendored impl does not expose this symbol").
        module.__getattr__ = lambda name, _t=target: getattr(_t, name)


class SglKernelNpuStubFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _VENDORED_ROUTES:
            return ModuleSpec(
                fullname,
                _VendoredRedirectLoader(_VENDORED_ROUTES[fullname]),
                is_package=False,
            )
        if fullname == "sgl_kernel_npu" or fullname.startswith("sgl_kernel_npu."):
            return ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def install() -> bool:
    """Install stub finder if the real sgl_kernel_npu is absent."""
    try:
        import sgl_kernel_npu  # noqa: F401
        return False
    except ImportError:
        pass
    if any(isinstance(f, SglKernelNpuStubFinder) for f in sys.meta_path):
        return True
    sys.meta_path.insert(0, SglKernelNpuStubFinder())
    return True


_installed = install()
