# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch distributed initialization for txda platform.

Patches applied:

  1.  ``_DEVICE_TO_DISTRIBUTED_BACKEND`` — add ``"txda": "tccl"``.
      Replaces ``parallel_state.py:1606``.

  2.  ``GroupCoordinator.__init__`` — inject txda device recognition so that
      ``self.device`` is ``txda:{local_rank}`` instead of ``cpu``.
      Replaces ``parallel_state.py:267-282``.

  3.  ``init_distributed_environment`` (srt) — force ``backend="tccl"``
      on txda so that ``torch.distributed.init_process_group`` uses the
      correct communicator library.
      Replaces ``parallel_state.py:1722``.

  4.  ``init_distributed_environment`` (multimodal_gen) — same backend
      override for the multimodal-gen codepath.
      Replaces ``multimodal_gen/.../parallel_state.py:234``.

  5.  ``GroupCoordinator.all_gather_into_tensor`` — route txda through
      the raw ``_all_gather_into_tensor`` path (same as npu/xpu), and
      set ``_is_txda`` module-level flag.
      Replaces ``parallel_state.py:65,818-823``.
"""

import functools

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("dist_init")

_patched = False
_originals = {}


# ── 1. _DEVICE_TO_DISTRIBUTED_BACKEND ────────────────────────────────────────


def _patch_backend_dict() -> None:
    """Add ``"txda": "tccl"`` to the distributed backend lookup dict."""
    try:
        import sglang.srt.distributed.parallel_state as _ps

        if "txda" not in _ps._DEVICE_TO_DISTRIBUTED_BACKEND:
            _ps._DEVICE_TO_DISTRIBUTED_BACKEND["txda"] = "tccl"
            _log.applied(
                "added 'txda' → 'tccl' to _DEVICE_TO_DISTRIBUTED_BACKEND"
            )
        _originals.setdefault("backend_dict", True)
    except Exception as exc:
        _log.failed("failed to patch backend dict: %s", exc)


# ── 2. GroupCoordinator.__init__ ─────────────────────────────────────────────


def _patch_group_coordinator_device() -> None:
    """Wrap ``GroupCoordinator.__init__`` to handle txda device detection.

    On txda, ``is_cuda_alike()`` returns False, so the original init falls
    to the ``else`` branch and sets ``self.device = torch.device("cpu")``.
    This breaks all downstream tensor operations.

    Strategy:
      - Temporarily make ``torch.cuda.is_available()`` return True and clear
        the ``is_cuda_alike`` LRU cache *before* calling the original init.
        This causes the init to take the ``is_cuda_alike()`` branch and set
        ``self.device`` to a cuda device (which ``torch_txda`` maps to txda
        hardware transparently).
      - After the init returns, replace the cuda device with the canonical
        txda device and refresh ``device_module``.
      - All distributed backend parameters (``torch_distributed_backend``)
        go through PlatformFL's ``get_torch_distributed_backend_str()`` which
        already returns ``"tccl"`` — no additional patching needed inside
        ``__init__``.
    """
    try:
        import sglang.srt.distributed.parallel_state as _ps
        import sglang.srt.utils.common as _srt_common

        GroupCoordinator = _ps.GroupCoordinator
        _orig_init = GroupCoordinator.__init__
        _originals.setdefault("gc_init", _orig_init)

        @functools.wraps(_orig_init)
        def _patched_gc_init(self, *args, **kwargs):
            if not _is_txda():
                return _orig_init(self, *args, **kwargs)

            _orig_cuda_avail = torch.cuda.is_available
            torch.cuda.is_available = lambda: True
            _srt_common.is_cuda_alike.cache_clear()

            try:
                _orig_init(self, *args, **kwargs)
            finally:
                torch.cuda.is_available = _orig_cuda_avail
                _srt_common.is_cuda_alike.cache_clear()

            local_rank = self.local_rank
            self.device = torch.device(f"txda:{local_rank}")
            self.device_module = torch.get_device_module(self.device)

        GroupCoordinator.__init__ = _patched_gc_init
        _log.applied(
            "GroupCoordinator.__init__ patched for txda device detection"
        )
    except Exception as exc:
        _log.failed(
            "failed to patch GroupCoordinator.__init__: %s", exc
        )


# ── 3. init_distributed_environment (srt) ───────────────────────────────────


def _patch_srt_init_dist() -> None:
    """Wrap srt init_distributed_environment to force tccl on txda.

    PlatformFL already returns ``"tccl"`` as the distributed backend via
    ``get_torch_distributed_backend_str()``, but some code-paths call
    ``init_distributed_environment`` with the default ``backend="nccl"``.
    This wrapper overrides the backend to ``"tccl"`` when txda is detected.
    """
    try:
        import sglang.srt.distributed.parallel_state as _ps

        _orig_fn = _ps.init_distributed_environment
        _originals.setdefault("srt_init_dist", _orig_fn)

        @functools.wraps(_orig_fn)
        def _patched_fn(*args, **kwargs):
            if _is_txda():
                kwargs["backend"] = "tccl"
            return _orig_fn(*args, **kwargs)

        _ps.init_distributed_environment = _patched_fn
        _log.applied(
            "srt init_distributed_environment patched → backend=tccl on txda"
        )
    except Exception as exc:
        _log.failed(
            "failed to patch srt init_distributed_environment: %s", exc
        )


# ── 4. init_distributed_environment (multimodal_gen) ────────────────────────


def _patch_multimodal_init_dist() -> None:
    """Wrap multimodal_gen init_distributed_environment to force tccl on txda.

    The multimodal-gen code has its own ``parallel_state`` module with a
    separate ``init_distributed_environment`` that does not share the backend
    lookup logic from the main srt module.
    """
    try:
        import sglang.multimodal_gen.runtime.distributed.parallel_state as _mps

        _orig_fn = _mps.init_distributed_environment
        _originals.setdefault("multimodal_init_dist", _orig_fn)

        @functools.wraps(_orig_fn)
        def _patched_fn(*args, **kwargs):
            if _is_txda():
                kwargs["backend"] = "tccl"
            return _orig_fn(*args, **kwargs)

        _mps.init_distributed_environment = _patched_fn
        _log.applied(
            "multimodal_gen init_distributed_environment patched → backend=tccl on txda"
        )
    except Exception as exc:
        _log.failed(
            "failed to patch multimodal_gen init_distributed_environment: %s",
            exc,
        )


# ── 5. GroupCoordinator.all_gather_into_tensor ─────────────────────────────


def _patch_all_gather_for_txda() -> None:
    """Set ``_is_txda`` flag and patch ``all_gather_into_tensor`` for txda.

    On txda, ``all_gather_into_tensor`` must route through the raw
    ``_all_gather_into_tensor`` path (same as npu/xpu) instead of the
    ``reg_all_gather_into_tensor`` custom-op path, which may not handle
    txda devices correctly.
    """

    try:
        import sglang.srt.distributed.parallel_state as _ps

        _txda_available = _is_txda()  # captured in closure below

        _orig_all_gather = _ps.GroupCoordinator.all_gather_into_tensor

        def _patched_all_gather(self, output, input_):
            if _txda_available:
                self._all_gather_into_tensor(output, input_)
            else:
                _ps.reg_all_gather_into_tensor(
                    output, input_, group_name=self.unique_name
                )

        if _orig_all_gather is _ps.GroupCoordinator.all_gather_into_tensor:
            _ps.GroupCoordinator.all_gather_into_tensor = _patched_all_gather
            _log.applied(
                "all_gather_into_tensor patched → direct path on txda"
            )
        else:
            _log.skipped("all_gather_into_tensor already patched")
    except Exception as exc:
        _log.failed("failed to patch all_gather_into_tensor: %s", exc)


# ── Public API ──────────────────────────────────────────────────────────────


def patch() -> None:
    """Apply all distributed-initialization patches for txda.

    Idempotent: safe to call multiple times.  Non-txda platforms are no-ops.
    """
    global _patched
    if _patched:
        return

    if not _is_txda():
        _log.skipped("txda not available — all dist patches skipped")
        return

    _patch_backend_dict()
    _patch_group_coordinator_device()
    _patch_srt_init_dist()
    _patch_multimodal_init_dist()
    _patch_all_gather_for_txda()

    _patched = True
