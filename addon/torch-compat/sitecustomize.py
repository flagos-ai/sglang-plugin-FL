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

"""Give a torch < 2.8 build the two surfaces sglang 0.5.18 imports at startup.

Both gaps are on the plain Qwen3 startup path, both are module-level imports
that no plugin can pre-empt, and both are imports only — the code behind them
is gated off on this platform (see "Why nothing is called" below).

1. ``torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`` and
   ``_cuda_endAllocateToPool``, imported by
   ``sglang.srt.distributed.device_communicators.pynccl_allocator``. torch 2.7
   spells the same two operations ``_cuda_beginAllocateToPool`` and
   ``_cuda_endAllocateCurrentStreamToPool``. This is torch's rename, not a
   substitute: upstream picks between the two spellings by version itself
   (``after_2_8_0`` in that file chooses ``torch._C._cuda_endAllocateToPool``
   on >= 2.8 and ``torch._C._cuda_endAllocateCurrentStreamToPool`` below it).

2. ``torch.distributed._symmetric_memory``, imported by
   ``sglang.srt.layers.logits_processor`` ->
   ``triton_symm_mem_ag``. The module's own import fails on torch < 2.8
   because ``torch._C._distributed_c10d._SymmetricMemory`` does not exist
   there. It is an NVLink multicast all-gather: this platform never runs it.

Why nothing is called
---------------------

Symmetric memory is off on this stack through sglang's own gate —
``pynccl_allocator.is_symmetric_memory_enabled()`` returns
``get_exec().comm.enable_symm_mem``, which is False without real symmetric
memory. So the aliased allocator entry points and the stubbed module names are
imported and never reached; a stub *call* raises, so a path that unexpectedly
reaches them fails loudly instead of computing something wrong.

Why ``sitecustomize``
---------------------

sglang runs its scheduler in a *spawned* worker, and that worker imports
``sglang.srt.managers.scheduler`` — taking in both surfaces above — before any
sglang plugin is loaded. A patch applied from the plugin is therefore never in
place early enough; the fix has to be on disk. This is the same ordering that
made ``addon/flashinfer-shim`` an installed package rather than a plugin patch.
``site`` imports this module at interpreter startup, so the patch is in place
in the worker too.

The hook only wraps the import of the ``sglang`` package: the patch runs
immediately before sglang's own code executes, so nothing heavier (importing
torch, say) is paid by interpreters that never import sglang. Both patches are
no-ops on torch >= 2.8.

Install scope
-------------

Publish to the vendor index of the backends that need it and list it in their
``deps_app``: this shim exists only for platforms whose SDK pins torch < 2.8.
"""

import importlib.machinery
import sys

# (name sglang imports, name this torch spells it as)
_CUDA_MEMORY_ALIASES = (
    ("_cuda_beginAllocateCurrentThreadToPool", "_cuda_beginAllocateToPool"),
    ("_cuda_endAllocateToPool", "_cuda_endAllocateCurrentStreamToPool"),
)

_SYMM_MEM_MODULE = "torch.distributed._symmetric_memory"

# Every name sglang reaches for on this module, taken from its call sites.
# Anything else raises AttributeError, so capability probes stay honest.
_SYMM_MEM_ALLOWED = (
    "empty",
    "enable_symm_mem_for_group",
    "get_buffer",
    "get_signal_pad_size",
    "multicast_ptr",
    "rendezvous",
    "set_signal_pad_size",
)

_MARKER = "_sglang_torch_compat_applied"
_installed = False


def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{_SYMM_MEM_MODULE}.{name} was called on a torch build without "
            f"symmetric memory — this call belongs behind "
            f"is_symmetric_memory_enabled()"
        )

    _raise.__name__ = name
    return _raise


def _apply():
    """Patch the two surfaces in the current interpreter. Idempotent."""
    module = sys.modules.get(__name__)
    if module is not None and getattr(module, _MARKER, False):
        return
    if module is not None:
        setattr(module, _MARKER, True)

    # 1. the renamed torch.cuda.memory entry points
    try:
        import torch.cuda.memory as _memory
    except Exception:  # no torch, or a build without the cuda module
        _memory = None
    if _memory is not None:
        for imported, present in _CUDA_MEMORY_ALIASES:
            if not hasattr(_memory, imported) and hasattr(_memory, present):
                setattr(_memory, imported, getattr(_memory, present))

    # 2. torch.distributed._symmetric_memory, when this torch cannot import it
    if _SYMM_MEM_MODULE not in sys.modules:
        try:
            import torch.distributed._symmetric_memory  # noqa: F401
        except ImportError:
            import types

            stub = types.ModuleType(_SYMM_MEM_MODULE)
            stub.__doc__ = (
                "sglang torch-compat shim: this torch build has no symmetric "
                "memory. Import-face only — every stubbed call raises."
            )
            for name in _SYMM_MEM_ALLOWED:
                setattr(stub, name, _unavailable(name))
            sys.modules[_SYMM_MEM_MODULE] = stub
            try:
                import torch.distributed as _distributed

                _distributed._symmetric_memory = stub
            except Exception:
                pass


class _ApplyBeforeExec:
    """Loader proxy: patch, then let the package's own code run."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        _apply()
        self._loader.exec_module(module)


class _SglangImportHook:
    """Wrap the import of the ``sglang`` package itself."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang" or fullname in sys.modules:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            return None
        if spec is not None and spec.loader is not None:
            spec.loader = _ApplyBeforeExec(spec.loader)
        return spec


def install():
    """Register the hook once, at interpreter start."""
    global _installed
    if _installed:
        return
    _installed = True
    sys.meta_path.insert(0, _SglangImportHook())
    if "sglang" in sys.modules:
        # Already imported (an embedded interpreter, or a test harness that
        # imported sglang before this module): patch now.
        _apply()


install()
