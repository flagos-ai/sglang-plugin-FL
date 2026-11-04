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

"""Let SGLang import without CUDA-only packages on the PPU empty runtime.

PPU reports ``is_cuda()``, so SGLang 0.5.18 takes its CUDA import paths, and
some of them import optional NVIDIA packages at module level with no
availability check:

- ``flashinfer``: ``layers/quantization/fp8_utils.py``.
- ``deep_gemm``: ``layers/attention/dsa_backend.py``, which the EAGLE/MTP worker
  imports while capturing draft CUDA graphs.

The empty runtime ships neither. For each missing package this guard serves a
placeholder package that fails only when one of its objects is actually called,
and forces SGLang's availability switch for it off so no code path selects it.
(DeepGEMM is also off by capability: PPU reports SM 80 and SGLang requires 90+.)

Finders are appended to ``sys.meta_path``, so an installed package always wins.
Installed from ``activate_platform()``, which runs during SGLang's platform
resolution, before any ``sglang.srt.layers`` module, in every process.
Per-package opt-out: ``SGLANG_FL_THEAD_FLASHINFER_STUB=0`` /
``SGLANG_FL_THEAD_DEEPGEMM_STUB=0``.
"""

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

# root package -> (opt-out env var, SGLang env overrides while the placeholder is active)
_PACKAGES = {
    # is_flashinfer_available() checks find_spec("flashinfer"), which the
    # placeholder satisfies; the env override makes it report False.
    "flashinfer": (
        "SGLANG_FL_THEAD_FLASHINFER_STUB",
        {"SGLANG_IS_FLASHINFER_AVAILABLE": "false"},
    ),
    "deep_gemm": (
        "SGLANG_FL_THEAD_DEEPGEMM_STUB",
        {"SGLANG_ENABLE_JIT_DEEPGEMM": "false"},
    ),
}


class _Unavailable:
    """Placeholder for any object of a missing package; usable at import time only."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        root = self._name.split(".", 1)[0]
        raise RuntimeError(
            f"{self._name} is unavailable: {root} is not installed on this "
            "platform (sglang_fl PPU placeholder)"
        )

    def __mro_entries__(self, bases):
        # Allow `class X(pkg.Foo)` at import time.
        return (object,)

    def __repr__(self) -> str:
        return f"<unavailable {self._name}>"


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self.__name__}.{attr}")


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path, target=None):
        if fullname != self.root and not fullname.startswith(self.root + "."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        module.__sglang_fl_placeholder__ = True
        return module

    def exec_module(self, module):
        pass


def _active_finder(root: str):
    for finder in sys.meta_path:
        if isinstance(finder, _PlaceholderFinder) and finder.root == root:
            return finder
    return None


def _installed(root: str) -> bool:
    for finder in sys.meta_path:
        if isinstance(finder, _PlaceholderFinder):
            continue
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is not None and find_spec(root, None) is not None:
            return True
    return False


def install_placeholder(root: str) -> bool:
    """Install the placeholder for ``root`` if it is missing. Returns True if active."""
    opt_out, env_overrides = _PACKAGES[root]
    if os.environ.get(opt_out, "1") == "0":
        return False
    if _active_finder(root) is not None:
        return True
    if _installed(root):
        return False

    sys.meta_path.append(_PlaceholderFinder(root))
    os.environ.update(env_overrides)
    logger.info("thead: %s not installed; import placeholder active", root)
    return True


def install_import_guards() -> list:
    """Install placeholders for every missing package. Returns the active roots."""
    return [root for root in _PACKAGES if install_placeholder(root)]
