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
- ``sgl_kernel``: the srt control path probes it lazily (e.g.
  ``layers/attention/triton_backend.py`` imports
  ``sgl_kernel.utils.is_arch_support_pdl``). The engine fails without a
  distribution answer for these imports (run 43845873116), but the
  Qwen3/Qwen3.6 CI matrix never enters a native sgl_kernel op
  (functional was fully green there).

The empty runtime ships none of them. For each missing package this guard
serves a placeholder package that fails only when one of its objects is
actually called, and forces SGLang's availability switch for it off so no
code path selects it. (DeepGEMM is also off by capability: PPU reports
SM 80 and SGLang requires 90+.)

``sgl_kernel`` carries two exact semantic exceptions on top of the generic
call-fails-loudly behavior (see ``_SEMANTIC_MEMBERS``):

- ``sgl_kernel.utils.is_arch_support_pdl()`` returns ``False`` — the real
  control-flow query; PPU has no PDL semantics.
- ``sgl_kernel.scalar_type`` does not exist (``AttributeError``), so
  SGLang's own ``hasattr``/try-import fallback selects its non-kernel
  path instead of receiving a placeholder object.

Nothing else is faked: no ``flash_mla``/``flash_attn`` implementations, no
``torch.ops.sgl_kernel`` registrations — any genuine native-kernel entry
fails loudly, which is the signal for switching to the real PPU
``sglang-kernel`` wheel fallback (docker/thead/kernel_fallback.sh).

Finders are appended to ``sys.meta_path``, so an installed package always wins;
the guard also skips a root when a real distribution answers ``find_spec``.
Installed from ``activate_platform()``, which runs during SGLang's platform
resolution, before any ``sglang.srt.layers`` module, in every process.
Per-package opt-out: ``SGLANG_FL_THEAD_FLASHINFER_STUB=0`` /
``SGLANG_FL_THEAD_DEEPGEMM_STUB=0`` / ``SGLANG_FL_THEAD_SGL_KERNEL_STUB=0``.
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
    # The placeholder serves no dist-info, so SGLang's sgl-kernel
    # wheel-version probe (entrypoints/engine.py, utils/common.py) cannot
    # see a version; the env override skips that probe for as long as the
    # placeholder is active. When the real PPU wheel fallback is installed,
    # the probe stays skipped for its older-than-NVIDIA-pin version (same
    # setting the validated images baked).
    "sgl_kernel": (
        "SGLANG_FL_THEAD_SGL_KERNEL_STUB",
        {"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "true"},
    ),
}

# Semantic members served by the placeholder, keyed by full dotted path.
#   - a callable/plain value: returned for the attribute (real semantics,
#     never an error);
#   - _ABSENT: the attribute raises AttributeError so SGLang's own
#     fallback logic (hasattr / try-import) selects its non-kernel path.
# Everything not listed here keeps the generic behavior: usable at import
# time, RuntimeError on the first real call.
_ABSENT = object()


def _is_arch_support_pdl() -> bool:
    """Real query, answered for real: PPU has no PDL (programmatic dependent
    launch) semantics, so SGLang must not add launch_pdl kwargs."""
    return False


_SEMANTIC_MEMBERS = {
    "sgl_kernel": {
        "sgl_kernel.utils.is_arch_support_pdl": _is_arch_support_pdl,
        # Deliberately absent (not a placeholder object): the engine probes
        # scalar_type to pick its sgl-kernel dtype path, and a usable-looking
        # placeholder here would steer it onto a path that cannot work.
        "sgl_kernel.scalar_type": _ABSENT,
    },
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
        key = f"{self.__name__}.{attr}"
        root = self.__name__.split(".", 1)[0]
        semantics = _SEMANTIC_MEMBERS.get(root, {})
        if key in semantics:
            member = semantics[key]
            if member is _ABSENT:
                raise AttributeError(attr)
            return member
        return _Unavailable(key)


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path, target=None):
        if fullname != self.root and not fullname.startswith(self.root + "."):
            return None
        # Members declared _ABSENT must not be importable either: an
        # `import sgl_kernel.scalar_type` (or the submodule fallback inside
        # `from sgl_kernel import scalar_type`) must fail, so SGLang's own
        # try-import fallback selects its non-kernel path.
        semantics = _SEMANTIC_MEMBERS.get(self.root, {})
        if semantics.get(fullname) is _ABSENT:
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
