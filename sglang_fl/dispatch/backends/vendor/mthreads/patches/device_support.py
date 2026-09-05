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

"""MUSA device-support monkey-patches on ``sglang.srt.utils.common``.

Upstream sglang detects MUSA by importing ``torchada``, a CUDA-alias layer
that routes ``torch.cuda.*`` onto MUSA.  The flagos mthreads runtime ships
``torch_musa`` instead (``torch.musa``, PrivateUse1, no CUDA alias), so
upstream ``is_musa()`` returns False and the capability helpers it gates on
(implemented through ``torch.cuda.*``) assert.  These patches:

  1. ``common.is_musa`` — recognise ``torch_musa`` in addition to ``torchada``.
  2. ``common.get_device_sm`` / ``get_device_capability`` /
     ``get_device_core_count`` / ``get_device_count`` — when MUSA is active
     but ``torch.cuda`` is not a live alias, route to ``torch.musa`` instead.

Already-imported copies of these names (``sglang.srt.utils`` re-exports and
module-level ``from ... import`` bindings) are rebound so the patch holds
regardless of import order.
"""

from __future__ import annotations

import functools
import logging
import sys

import torch

logger = logging.getLogger(__name__)

_patched = False
_originals: dict[str, object] = {}
_replacements: dict[str, object] = {}


@functools.lru_cache(maxsize=1)
def _is_musa() -> bool:
    """True when a MUSA torch stack is importable and live.

    Accepts both upstream's ``torchada`` CUDA-alias layer and the flagos
    runtime's ``torch_musa`` (PrivateUse1) plugin.
    """
    for pkg in ("torchada", "torch_musa"):
        try:
            __import__(pkg)
        except ImportError:
            continue
        if getattr(torch.version, "musa", None) is not None:
            return True
        if hasattr(torch, "musa"):
            try:
                if torch.musa.is_available():
                    return True
            except Exception:
                # is_available() can raise in forked children (MUSA cannot
                # re-initialise); only the version attribute remains usable.
                pass
    return False


def _use_musa_direct() -> bool:
    """MUSA active but no CUDA alias — helpers must call torch.musa directly."""
    return _is_musa() and not torch.cuda.is_available()


# ──────────────────────────────────────────────────────────────────────────────
# Patched helper implementations (MUSA-direct branches)
# ──────────────────────────────────────────────────────────────────────────────


def _get_device_sm() -> int:
    if _use_musa_direct():
        major, minor = torch.musa.get_device_capability()
        return major * 10 + minor
    return _originals["get_device_sm"]()  # type: ignore[no-any-return]


def _get_device_capability(device_id: int = 0):
    if _use_musa_direct():
        return torch.musa.get_device_capability(device_id)
    return _originals["get_device_capability"](device_id)


def _get_device_core_count(device_id: int = 0) -> int:
    if _use_musa_direct():
        return torch.musa.get_device_properties(device_id).multi_processor_count
    return _originals["get_device_core_count"](device_id)


@functools.lru_cache(maxsize=1)
def _get_device_count() -> int:
    if _use_musa_direct():
        return torch.musa.device_count()
    return _originals["get_device_count"]()  # type: ignore[no-any-return]


def _init_cublas():
    """Warmup matmul that initialises the vendor BLAS library.

    Upstream hardcodes ``device="cuda"`` (common.init_cublas), written against
    the torchada CUDA-alias MUSA stack.  The flagos runtime ships torch_musa
    (PrivateUse1, no CUDA alias), so route the warmup to ``torch.musa`` when
    CUDA is not live.  On real CUDA this replacement is never selected.
    """
    if _use_musa_direct():
        dtype = torch.float16
        a = torch.ones((16, 16), dtype=dtype, device="musa")
        b = torch.ones((16, 16), dtype=dtype, device="musa")
        return a @ b
    return _originals["init_cublas"]()  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────────────
# Patch application
# ──────────────────────────────────────────────────────────────────────────────


def _rebind_importers() -> None:
    """Point already-imported copies of patched names at the replacements."""
    for mod in tuple(sys.modules.values()):
        for name, orig in _originals.items():
            if getattr(mod, name, None) is orig:
                setattr(mod, name, _replacements[name])


def patch_device_support() -> None:
    """Apply MUSA device-support patches.  Idempotent, no-op off-MUSA."""
    global _patched
    if _patched:
        return
    _patched = True

    if not _is_musa():
        logger.info("MUSA torch stack not detected; device-support patch skipped")
        return

    from sglang.srt.utils import common

    # is_musa supersedes upstream's torchada-only body; no original fallback.
    _originals["is_musa"] = common.is_musa
    _replacements["is_musa"] = _is_musa
    common.is_musa = _is_musa

    # Results cached under the old (torchada-only) detection are stale now.
    for name in ("get_device", "get_device_count", "get_device_sm"):
        fn = getattr(common, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()

    # Capability/count helpers: route to torch.musa only when torch.cuda is
    # not an alias (torch_musa runtime); upstream behaviour is kept otherwise.
    for name, replacement in (
        ("get_device_sm", _get_device_sm),
        ("get_device_capability", _get_device_capability),
        ("get_device_core_count", _get_device_core_count),
        ("get_device_count", _get_device_count),
        ("init_cublas", _init_cublas),
    ):
        orig = getattr(common, name, None)
        if orig is None:
            logger.warning("common.%s not found; skipping its MUSA routing", name)
            continue
        _originals[name] = orig
        _replacements[name] = replacement
        setattr(common, name, replacement)

    _rebind_importers()
    logger.info(
        "MUSA device-support patches applied "
        "(is_musa + torch.musa capability routing)"
    )
