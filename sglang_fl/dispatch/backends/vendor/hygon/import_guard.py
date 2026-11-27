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

"""Let SGLang import when HCU ships no sgl-kernel or FlashInfer wheel.

SGLang 0.5.18 imports modules below ``sgl_kernel`` and ``flashinfer`` while
loading layers that HCU later dispatches to FlagOS implementations.  The HCU
Runtime provides neither distribution.  During Hygon platform discovery this
module appends narrowly scoped placeholder finders so real installed packages
always win.

The placeholder contract intentionally matches the formerly image-written
stubs:

* ``sgl_kernel.utils.is_arch_support_pdl()`` returns ``False``;
* ``sgl_kernel.scalar_type`` is never fabricated, preserving SGLang's
  ``ImportError`` fallback;
* every other unknown object may be imported but raises ``RuntimeError`` when
  called; and
* no distribution metadata is fabricated.  The SGLang kernel-version probe is
  skipped only while the ``sgl_kernel`` placeholder is active.

Installation is process-local and idempotent.  Engine workers and descendants
must each execute Hygon platform activation, as they already do for other
vendor early hooks.  Set ``SGLANG_FL_HYGON_SGL_KERNEL_STUB=0`` or
``SGLANG_FL_HYGON_FLASHINFER_STUB=0`` to disable one placeholder.
"""

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

_PACKAGES = {
    "sgl_kernel": "SGLANG_FL_HYGON_SGL_KERNEL_STUB",
    "flashinfer": "SGLANG_FL_HYGON_FLASHINFER_STUB",
}
_SCALAR_TYPE = "sgl_kernel.scalar_type"
_SKIP_VERSION_CHECK = "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"


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
            f"{self._name} is unavailable: {root} is not installed on HCU "
            "(sglang_fl placeholder)"
        )

    def __mro_entries__(self, bases):
        return (object,)

    def __repr__(self) -> str:
        return f"<unavailable {self._name}>"


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self.__name__}.{attr}")


def _pdl_is_unsupported() -> bool:
    return False


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path, target=None):
        if fullname == _SCALAR_TYPE:
            return None
        if fullname != self.root and not fullname.startswith(self.root + "."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        module.__sglang_fl_placeholder__ = True
        if spec.name == "sgl_kernel.utils":
            module.is_arch_support_pdl = _pdl_is_unsupported
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
    if root == "sgl_kernel":
        os.environ[_SKIP_VERSION_CHECK] = "1"
    logger.info("hygon: %s not installed; import placeholder active", root)
    return True


def install_import_guards() -> list[str]:
    """Install guards for missing packages and return the active roots."""
    return [root for root in _PACKAGES if install_placeholder(root)]
