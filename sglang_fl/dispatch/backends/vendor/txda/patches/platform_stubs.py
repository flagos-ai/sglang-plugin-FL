# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch ``sys.modules`` with stub no-op modules for platform-specific
dependencies that are unavailable on non‑CUDA platforms (txda, CPU, etc.).

On txda / CPU, ``sgl_kernel`` and ``flashinfer`` cannot be imported because
they rely on ``torch.cuda`` and compiled CUDA kernels.  However SGLang's
module-level code may attempt to import these symbols at various points
during runtime (lazy imports, type annotations, etc.).  Without these
stub modules every such import would raise ``ImportError`` and crash the
server.

We inject ``_StubModule`` instances into ``sys.modules`` for a known set
of module names.  The stubs silently propagate attribute access and are
callable, so any code path that references ``flashinfer.prefill.xxx(...)``
simply returns another stub instead of failing.  The actual call sites are
replaced by the plugin's dispatch system before any real work happens, so
these stubs are only needed for the import chain to survive.

Debug logging:
  When a stub op is actually *called* (i.e. ``stub_module.some_fn()``),
  a ``DEBUG``-level log is emitted showing the full dotted path of the call.
  This helps identify code paths that are unexpectedly reaching stub ops
  instead of the intended dispatch backend.  Set ``SGLANG_FL_LOG_LEVEL=DEBUG``
  to see these messages.

Patches applied:
  1. ``sys.modules`` — insert ``_StubModule`` instances for every module
     listed in ``_PLATFORM_STUB_MODULES`` that has not already been imported.

Controlled by ``SGLANG_FL_STUB_MODULES_EXTRA`` (optional): comma-separated
list of additional module names to stub out beyond the built-in set.
"""

import os
import sys
import types

from sglang_fl.dispatch.backends.vendor.txda.patches._logger import patch_logger
try:
    import torch_txda  # noqa: F401
    from torch_txda import transfer_to_txda
except ImportError:
    pass


_log = patch_logger("platform_stubs")

_patched = False

# ---------------------------------------------------------------------------
# Built-in list of modules to stub-out on non-CUDA platforms.
# These are all modules whose import-time code touches torch.cuda, CUDA
# kernels, or other GPU-only facilities not available on txda/CPU.
# ---------------------------------------------------------------------------
_PLATFORM_STUB_MODULES = frozenset(
    {
        "sgl_kernel",
        "sgl_kernel.elementwise",
        "sgl_kernel.flash_attn",
        "sgl_kernel.flash_mla",
        "sgl_kernel.kvcacheio",
        "sgl_kernel.mamba",
        "sgl_kernel.quantization",
        "sgl_kernel.scalar_type",
        "sgl_kernel.sparse_flash_attn",
        "sgl_kernel.speculative",
        "flashinfer",
        "flashinfer.autotuner",
        "flashinfer.cascade",
        "flashinfer.comm",
        "flashinfer.decode",
        "flashinfer.fused_moe",
        "flashinfer.gdn_decode",
        "flashinfer.gdn_kernels",
        "flashinfer.gdn_prefill",
        "flashinfer.gemm",
        "flashinfer.mamba",
        "flashinfer.norm",
        "flashinfer.prefill",
        "flashinfer.sampling",
        "flashinfer.utils",
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        # "torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool"
    }
)


# ---------------------------------------------------------------------------
# Stub classes
# ---------------------------------------------------------------------------


class _StubLoader:
    """Loader that satisfies ``importlib`` without actually importing anything.

    Python's import system requires ``module.__spec__.loader`` to be non‑None
    for the module to be treated as "fully loaded".  ``importlib.util.find_spec``
    also raises ``ValueError`` when ``__spec__`` is ``None``.

    A non‑None loader also prevents ``importlib._bootstrap._find_and_load``
    from falling through to filesystem discovery, which would find and execute
    the real ``.py`` file and crash on non‑CUDA platforms.
    """

    def create_module(self, spec):
        return None  # let importlib create the module normally

    def exec_module(self, module):
        pass  # already initialized


class _StubObj:
    """Callable stub that propagates attribute access recursively.

    Supports the context-manager protocol so that
    ``with stub_callable(...):`` works without error.

    When *called* (i.e. someone treats a stub module attribute as a
    function), a debug log is emitted recording the full dotted path of
    the call site.  Only the **first** call per path is logged to avoid
    flooding the output.
    """

    _call_logged: set = set()  # class-level dedup of already-logged call paths

    def __init__(self, path: str = "") -> None:
        object.__setattr__(self, "_path", path)

    def __call__(self, *args: object, **kwargs: object) -> "_StubObj":
        path = self._path or "<anonymous>"
        if path not in _StubObj._call_logged:
            _StubObj._call_logged.add(path)
            _log.debug(
                "stub op called: %s(args=%s, kwargs=%s)",
                path,
                _summarize(args),
                _summarize(kwargs),
            )
        return self

    def __getattr__(self, name: str) -> "_StubObj":
        if name.startswith("__"):
            raise AttributeError(name)
        path = self._path or ""
        child_path = f"{path}.{name}" if path else name
        val = _StubObj(path=child_path)
        object.__setattr__(self, name, val)
        return val

    def __enter__(self) -> "_StubObj":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _StubModule(types.ModuleType):
    """Module subclass whose every attribute access returns a :class:`_StubObj`.

    The stub objects carry the full dotted path (e.g.
    ``"flashinfer.prefill.single_prefill_with_kv_cache"``) so that debug
    logs can pinpoint which missing symbol was referenced.

    **Important**: the ``__spec__.loader`` must be a real (non‑None) object
    so that ``importlib.util.find_spec`` doesn't raise and the import
    machinery treats the module as fully loaded instead of searching the
    filesystem for the real ``.py`` file.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Provide a proper spec with a stub loader so importlib treats
        # this module as fully loaded and filesystem discovery is skipped.
        try:
            from importlib.machinery import ModuleSpec

            self.__spec__ = ModuleSpec(name, loader=_StubLoader())
        except ImportError:
            self.__spec__ = None

    def __getattr__(self, name: str) -> _StubObj:
        if name.startswith("__"):
            raise AttributeError(name)
        val = _StubObj(path=f"{self.__name__}.{name}")
        setattr(self, name, val)
        return val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize(val: object, max_len: int = 160) -> str:
    """Short string representation of a value for debug-logging call args."""
    s = repr(val)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _resolve_module_names() -> list[str]:
    """Return the final list of module names to stub out.

    Merges the built-in ``_PLATFORM_STUB_MODULES`` set with any extra names
    from the ``SGLANG_FL_STUB_MODULES_EXTRA`` environment variable.
    """
    names = sorted(_PLATFORM_STUB_MODULES)

    extra_env = os.environ.get("SGLANG_FL_STUB_MODULES_EXTRA", "").strip()
    if extra_env:
        extras = [n.strip() for n in extra_env.split(",") if n.strip()]
        names.extend(extras)
        _log.info("loaded %d extra stub module names from env", len(extras))

    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch() -> None:
    """Inject stub modules into ``sys.modules`` for all target names that
    have not yet been imported.

    Idempotent: safe to call multiple times.  Already-imported (real)
    modules are never replaced.
    """
    global _patched
    if _patched:
        return

    mod_names = _resolve_module_names()
    injected: list[str] = []
    skipped: list[str] = []

    for mod_name in mod_names:
        if mod_name in sys.modules:
            skipped.append(mod_name)
            continue
        sys.modules[mod_name] = _StubModule(mod_name)
        injected.append(mod_name)

    if injected:
        _log.applied(
            "injected %d stub modules: %s",
            len(injected),
            ", ".join(injected),
        )
    else:
        _log.skipped("no modules to inject — all targets already in sys.modules")

    if skipped:
        _log.info(
            "skipped %d already-loaded real modules: %s",
            len(skipped),
            ", ".join(skipped),
        )

    _patched = True
