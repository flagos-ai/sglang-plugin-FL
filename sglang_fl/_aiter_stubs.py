# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shim aiter imports on HIP platforms that ship no aiter wheel.

On ROCm-based stacks (Hygon DTK, real AMD ROCm), ``torch.version.hip`` is set, so
sglang's quantization layer treats the platform as HIP and some Quark /
compressed-tensors scheme modules do an unconditional module-level
``from aiter...`` import (no try/except). sglang-plugin-FL loads sglang's
quantization package during ``load_plugin`` (dispatch hook construction), so a
HIP box without an ``aiter`` wheel crashes the plugin import itself — well
before any quantization scheme is actually selected.

The stub layer pre-seeds ``sys.modules`` with flat stand-in modules for the full
aiter import surface found in the sglang 0.5.18 wheel (see the frozenset
below), so those imports resolve. The stubs are inert objects: they answer
attribute access and calls, and never touch hardware, so a scheme that ends up
running on them would fail loudly (not silently) at the point a real value is
needed — no dunder probing is ever faked.

Installation is gated on the environment, not the vendor: only when HIP is
active AND no real ``aiter`` is importable are stubs installed. A genuine
aiter wheel (real AMD) therefore keeps full functionality and the stubs simply
never engage.
"""

import importlib.util
import logging
import sys
import types

logger = logging.getLogger(__name__)

# The full aiter import surface of the sglang 0.5.18 wheel's quantization tree.
# Flat coverage is required: each intermediate parent package is also seeded,
# because a stub parent without ``__path__`` cannot drive submodule traversal.
_AITER_STUB_MODULES = frozenset(
    {
        "aiter",
        "aiter.ops",
        "aiter.ops.shuffle",
        "aiter.ops.triton",
        "aiter.ops.triton.gemm",
        "aiter.ops.triton.gemm.fused",
        "aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat",
        "aiter.ops.triton.gemm_afp4wfp4",
        "aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic",
        "aiter.ops.triton.gemm_a8w8_blockscale",
        "aiter.ops.triton.quant",
        "aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant",
        "aiter.ops.triton.fused_mxfp4_quant",
        "aiter.utility",
        "aiter.utility.fp4_utils",
        "aiter.tuned_gemm",
    }
)


class _StubLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass


class _StubObj:
    """Attribute-and-call surrogate; never equal to a real kernel result."""

    _call_logged: set = set()

    def __init__(self, path: str = "") -> None:
        object.__setattr__(self, "_path", path)

    def __call__(self, *args, **kwargs):
        path = self._path or "<anonymous>"
        if path not in _StubObj._call_logged:
            _StubObj._call_logged.add(path)
            logger.debug("stub op called: %s(...)", path)
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        path = self._path or ""
        child_path = f"{path}.{name}" if path else name
        val = _StubObj(path=child_path)
        object.__setattr__(self, name, val)
        return val

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _StubModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        try:
            from importlib.machinery import ModuleSpec

            self.__spec__ = ModuleSpec(name, loader=_StubLoader())
        except ImportError:
            self.__spec__ = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        val = _StubObj(path=f"{self.__name__}.{name}")
        setattr(self, name, val)
        return val


_installed = False


def install_aiter_stubs_if_needed() -> bool:
    """Install the aiter stub tree when the platform is HIP without aiter.

    No-op unless ``torch.version.hip`` is set and ``aiter`` is not importable,
    which is exactly the configuration that would crash the quantization import
    chain. Returns True when stubs were (or already had been) installed.
    """
    global _installed
    if _installed:
        return True

    torch = sys.modules.get("torch")
    if torch is None or getattr(torch.version, "hip", None) is None:
        return False

    if importlib.util.find_spec("aiter") is not None:
        logger.info("aiter is importable — not installing aiter import stubs")
        return False

    installed = 0
    for name in sorted(_AITER_STUB_MODULES):
        if name in sys.modules:
            continue
        sys.modules[name] = _StubModule(name)
        installed += 1
    logger.info("installed aiter import stubs for HIP platform (%d modules)", installed)

    _installed = True
    return True
