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

"""Guard the MUSA-branch ``flash_attn_interface`` import on sglang internals.

Upstream sglang 0.5.18 gates two import sites on MUSA:

  - ``sglang.srt.layers.attention.vision`` binds ``flash_attn_varlen_func``
    from ``flash_attn_interface`` at module level whenever ``is_musa()`` is
    live (vision.py:66-67), and that module is pulled in model-independently
    (http_server -> openai -> realtime ASR -> qwen3_omni_moe).
  - ``sglang.srt.hardware_backend.musa.attention.flashattention_backend``
    imports ``flash_attn_varlen_func`` / ``flash_attn_with_kvcache`` /
    ``get_scheduler_metadata`` from the package at module top (the ``fa3``
    text attention backend for MUSA).

The flagos mthreads runtime ships ``torch_musa`` but no ``flash_attn_interface``
(Moore Threads' flash-attention v3 interface wheel), so with MUSA detection
correct these imports raise ``ModuleNotFoundError`` before the engine starts.
This patch injects a stub ``flash_attn_interface`` into ``sys.modules`` when
the real package is absent, letting those modules import.  The stub's callables
raise ``NotImplementedError`` if actually invoked, so any MUSA code path that
needs the flash-attention v3 kernels fails loudly at the call site instead of
producing wrong numerics.

The stub is only installed when (a) the MUSA torch stack is live and (b) the
real package cannot be imported, so upstream-style MUSA runtimes that do carry
``flash_attn_interface`` are untouched.
"""

from __future__ import annotations

import logging
import sys
import types

from sglang_fl.dispatch.backends.vendor.mthreads.patches.device_support import (
    _is_musa,
)

logger = logging.getLogger(__name__)

_patched = False

_UNSUPPORTED_MSG = (
    "flash_attn_interface.{name} is unavailable on the flagos mthreads runtime "
    "(no Moore Threads flash-attention v3 interface wheel is installed).  A "
    "MUSA code path that requires it was reached; this attention backend is "
    "not supported here."
)


class _UnavailableCallable:
    """Callable that raises ``NotImplementedError`` when invoked."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<unavailable flash_attn_interface.{self._name}>"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED_MSG.format(name=self._name))


class _StubModule(types.ModuleType):
    """Module whose every attribute access yields a raising callable.

    ``from flash_attn_interface import <symbol>`` succeeds for any symbol;
    the bound name fails only when actually called.
    """

    def __getattr__(self, name: str) -> _UnavailableCallable:
        if name.startswith("__"):
            raise AttributeError(name)
        callable_ = _UnavailableCallable(name)
        setattr(self, name, callable_)
        return callable_


def patch_flash_attn_interface() -> None:
    """Insert the ``flash_attn_interface`` stub when MUSA needs it and it is
    missing.  Idempotent; no-op when the real package is importable.
    """
    global _patched
    if _patched:
        return
    _patched = True

    if not _is_musa():
        logger.info("MUSA not active; flash_attn_interface guard skipped")
        return

    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        pass
    else:
        logger.info(
            "flash_attn_interface is importable; no stub needed"
        )
        return

    if "flash_attn_interface" in sys.modules:
        logger.info("flash_attn_interface already present in sys.modules")
        return

    stub = _StubModule("flash_attn_interface")
    stub.__doc__ = (
        "FlagOS mthreads stub: the real flash_attn_interface is not installed; "
        "imports survive, calls raise NotImplementedError."
    )
    sys.modules["flash_attn_interface"] = stub
    logger.info(
        "MUSA runtime has no flash_attn_interface; installed raising stub "
        "so sglang's flash_attn_interface import sites survive"
    )
