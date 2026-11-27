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

"""Let SGLang import when the operations Runtime ships no ``sgl_kernel_npu``.

SGLang 0.5.18 imports modules below ``sgl_kernel_npu`` module-level while
loading the NPU attention backends, the MoE top-k and the GDN kernel
dispatch. The Ascend operations Runtime provides no such wheel. During Ascend
platform discovery (``early.py``) this module appends a narrowly scoped
placeholder finder so a real installed ``sgl_kernel_npu`` always wins.

The placeholder contract intentionally matches the Hygon guard this plugin
already ships:

* import success only -- no distribution metadata is fabricated;
* the operators the CI model matrix calls get real implementations from
  ``sgl_kernel_npu_fallbacks`` (plain PyTorch or SGLang's own Triton kernels)
  attached to their placeholder submodules;
* every other unknown object may be imported but raises ``RuntimeError`` when
  called; and
* the symbols SGLang selects through ``try/except ImportError`` stay ABSENT:
  importing them raises the ``ImportError`` SGLang expects, so its generic
  fallback takes over instead of binding a placeholder object.

Installation is process-local and idempotent. Engine workers and descendants
must each execute Ascend platform activation, as they already do for the
other vendor early hooks. Set ``SGLANG_FL_ASCEND_SGL_KERNEL_NPU_STUB=0`` to
disable the placeholder.
"""

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

_PACKAGES = {
    "sgl_kernel_npu": "SGLANG_FL_ASCEND_SGL_KERNEL_NPU_STUB",
}

# module path -> attribute names the placeholder must NOT answer. The import
# system then raises the ImportError these call sites select fallbacks with
# (rotary fast path, fused QKV RMSNorm+RoPE), instead of binding a placeholder
# object that would only fail later, at call time.
_ABSENT_ATTRIBUTES = {
    "sgl_kernel_npu.norm": (
        "fused_rope_qk_mqa",
        "split_qkv_rmsnorm_rope_pos_cache_half_npu",
    ),
}

# Leaf module paths kept absent, derived from _ABSENT_ATTRIBUTES: the finder
# must not answer for them, so ``from sgl_kernel_npu.norm import fused_rope_qk_mqa``
# raises the ImportError the call sites select fallbacks with.
_ABSENT_MODULE_PATHS = {
    f"{parent}.{name}" for parent, names in _ABSENT_ATTRIBUTES.items() for name in names
}


class _Unavailable:
    """An import-time placeholder that fails loudly when executed."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        root = self._name.split(".", 1)[0]
        raise RuntimeError(
            f"{self._name} is unavailable: {root} is not installed on the "
            "operations Runtime (sglang_fl placeholder)"
        )

    def __mro_entries__(self, bases):
        return (object,)

    def __repr__(self) -> str:
        return f"<unavailable {self._name}>"


class _PlaceholderModule(types.ModuleType):
    def __init__(self, name: str, absent: tuple[str, ...] = ()):
        super().__init__(name)
        self._absent = absent

    def __getattr__(self, attr: str):
        if attr in self._absent:
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {attr!r} "
                "(kept absent so SGLang's ImportError fallback takes over)"
            )
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self.__name__}.{attr}")


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path, target=None):
        if fullname in _ABSENT_MODULE_PATHS:
            return None
        if fullname != self.root and not fullname.startswith(self.root + "."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        from sglang_fl.dispatch.backends.vendor.ascend import sgl_kernel_npu_fallbacks

        module = _PlaceholderModule(spec.name, _ABSENT_ATTRIBUTES.get(spec.name, ()))
        module.__path__ = []
        module.__sglang_fl_placeholder__ = True
        fallbacks = sgl_kernel_npu_fallbacks.SGL_KERNEL_NPU_FALLBACKS.get(spec.name, {})
        for name, implementation in fallbacks.items():
            setattr(module, name, implementation)
        return module

    def exec_module(self, module):
        pass


def _active_finder(root: str):
    for finder in sys.meta_path:
        if isinstance(finder, _PlaceholderFinder) and finder.root == root:
            return finder
    return None


def _installed(root: str) -> bool:
    """Check all real importers without allowing our placeholders to answer."""
    for finder in sys.meta_path:
        if isinstance(finder, _PlaceholderFinder):
            continue
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        try:
            if find_spec(root, None) is not None:
                return True
        except (AttributeError, ImportError, ModuleNotFoundError):
            continue
    return False


def install_placeholder(root: str) -> bool:
    """Install a missing-package placeholder; return whether it is active."""
    opt_out = _PACKAGES[root]
    if os.environ.get(opt_out, "1") == "0":
        return False
    if _active_finder(root) is not None:
        return True
    if _installed(root):
        return False

    sys.meta_path.append(_PlaceholderFinder(root))
    logger.info("ascend: %s not installed; import placeholder active", root)
    return True


def install_import_guards() -> list[str]:
    """Install guards for missing packages and return the active roots."""
    return [root for root in _PACKAGES if install_placeholder(root)]
