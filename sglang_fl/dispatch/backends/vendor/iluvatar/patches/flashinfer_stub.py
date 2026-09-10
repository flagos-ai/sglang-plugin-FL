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

"""Iluvatar: keep the is_cuda-gated flashinfer imports importable.

corex torch is CUDA-alias — `torch.version.cuda` is set, so sglang's
`is_cuda()` is True on iluvatar even though nothing CUDA-specific is usable
there. The corex platform has no flashinfer build (no vendor wheel, unlike
metax), but three module-level imports sit under a bare `if is_cuda()` with no
availability check, so they fire on paths that never touch the CUDA variant.
The first one is fatal on every serve — ServerArgs -> model_config ->
quantization -> fp8 -> fp8_utils runs before any model is loaded:

    File ".../sglang/srt/layers/quantization/fp8_utils.py", line 188
      from flashinfer import bmm_fp8 as _raw_bmm_fp8_batched
    ModuleNotFoundError: No module named 'flashinfer'

Registering stub modules keeps those imports working. The allowlist below is
deliberately explicit: `__getattr__` raises AttributeError for any name not in
it, so `hasattr(flashinfer_x, y)`-style capability probes elsewhere in sglang
still answer honestly instead of seeing a stub and taking a CUDA path. A name
outside the list still fails loudly at import (today's behaviour) — extend the
list when a new path needs a symbol, which keeps the set auditable.

`is_flashinfer_available()` must keep returning False: it selects the sampling
backend, the attention backend and the reported `sampling_backend`. Its
definition is `SGLANG_IS_FLASHINFER_AVAILABLE and find_spec("flashinfer") and
is_cuda()`, and a registered stub makes the find_spec term succeed — so the env
var is what holds it down, and it is defaulted here because a serve that lost
it would otherwise route sampling into these stubs.
"""

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

# module -> symbols sglang imports from it at module level under `if is_cuda`.
_ALLOWLIST = {
    "flashinfer": ("bmm_fp8",),
    "flashinfer.sampling": (
        "min_p_sampling_from_probs",
        "top_k_top_p_sampling_from_probs",
    ),
    "flashinfer.prefill": ("cudnn_batch_prefill_with_kv_cache",),
}

_patched = False


class _StubLoader:
    """Loader that satisfies importlib without executing anything.

    `module.__spec__.loader` must be non-None or `importlib.util.find_spec`
    raises instead of returning, which would turn every availability probe
    into a crash rather than a False.
    """

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass


def _unavailable(qualname):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{qualname} was called on iluvatar, but there is no flashinfer "
            f"build for corex — this call belongs behind "
            f"is_flashinfer_available()"
        )

    _raise.__name__ = qualname.rsplit(".", 1)[-1]
    return _raise


class _StubModule(types.ModuleType):
    """Stub whose attributes are limited to this module's allowlist."""

    def __init__(self, name, allowed):
        super().__init__(name)
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=_StubLoader())
        # Mark every stub as a package so a submodule import that slipped past
        # the sys.modules pre-registration resolves as a package attribute
        # rather than failing with "not a package".
        self.__path__ = []
        object.__setattr__(self, "_allowed", frozenset(allowed))

    def __getattr__(self, name):
        if name in self._allowed:
            return _unavailable(f"{self.__name__}.{name}")
        raise AttributeError(
            f"module {self.__name__!r} has no attribute {name!r} "
            f"(iluvatar flashinfer stub: only {sorted(self._allowed)} is stubbed)"
        )


def patch_flashinfer_stub() -> None:
    """Register the allowlisted flashinfer stubs and pin the availability env."""
    global _patched
    if _patched:
        return
    _patched = True

    if importlib.util.find_spec("flashinfer") is not None:
        # A real build is importable — nothing to stand in for.
        logger.info("flashinfer stub skipped: a real flashinfer is importable")
        return

    os.environ.setdefault("SGLANG_IS_FLASHINFER_AVAILABLE", "false")

    # Parents before children so `from flashinfer.sampling import x` resolves
    # through the already-registered module.
    for name in sorted(_ALLOWLIST):
        if name in sys.modules:
            continue
        module = _StubModule(name, _ALLOWLIST[name])
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, module)

    logger.info(
        "flashinfer stub installed for iluvatar: %s (SGLANG_IS_FLASHINFER_AVAILABLE=%s)",
        sorted(_ALLOWLIST),
        os.environ.get("SGLANG_IS_FLASHINFER_AVAILABLE"),
    )


patch_flashinfer_stub()
