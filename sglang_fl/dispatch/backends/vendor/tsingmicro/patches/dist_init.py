# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed-group fixes for txda (SGLang 0.5.18).

Patches applied:

  1.  ``GroupCoordinator.__init__`` — set ``self.device`` to
      ``txda:{local_rank}``. SGLang picks the group device from
      ``is_cuda_alike()`` (``torch.version.cuda`` based), which is False on the
      TXDA torch build, so it falls back to ``torch.device("cpu")``.

  2.  ``GroupCoordinator.all_gather_into_tensor`` — call the raw
      ``_all_gather_into_tensor`` on txda (as SGLang does for npu) instead of
      the ``reg_all_gather_into_tensor`` custom-op path.

Not needed on 0.5.18 (removed): the distributed backend ("tccl") comes from
``PlatformFL.get_torch_distributed_backend_str()`` via
``get_default_distributed_backend``; multimodal_gen picks its backend from its
own ``current_platform``.
"""

import functools

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("dist_init")

_patched = False
_originals = {}


# ── 1. GroupCoordinator.__init__ ─────────────────────────────────────────────


def _patch_group_coordinator_device() -> None:
    """Wrap ``GroupCoordinator.__init__``: replace the cpu fallback device with txda."""
    try:
        import sglang.srt.distributed.parallel_state as _ps

        GroupCoordinator = _ps.GroupCoordinator
        _orig_init = GroupCoordinator.__init__
        _originals.setdefault("gc_init", _orig_init)

        @functools.wraps(_orig_init)
        def _patched_gc_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if _is_txda():
                self.device = torch.device(f"txda:{self.local_rank}")
                self.device_module = torch.get_device_module(self.device)

        GroupCoordinator.__init__ = _patched_gc_init
        _log.applied("GroupCoordinator.__init__ patched -> device txda:{local_rank}")
    except Exception as exc:  # noqa: BLE001
        _log.failed("failed to patch GroupCoordinator.__init__: %s", exc)


# ── 2. GroupCoordinator.all_gather_into_tensor ─────────────────────────────


def _patch_all_gather_for_txda() -> None:
    """Route ``all_gather_into_tensor`` through the raw path on txda."""
    try:
        import sglang.srt.distributed.parallel_state as _ps

        def _patched_all_gather(self, output, input_):
            self._all_gather_into_tensor(output, input_)

        _ps.GroupCoordinator.all_gather_into_tensor = _patched_all_gather
        _log.applied("all_gather_into_tensor patched -> direct path on txda")
    except Exception as exc:  # noqa: BLE001
        _log.failed("failed to patch all_gather_into_tensor: %s", exc)


# ── Public API ──────────────────────────────────────────────────────────────


def patch() -> None:
    """Apply the distributed-group patches for txda. Idempotent; no-op elsewhere."""
    global _patched
    if _patched:
        return

    if not _is_txda():
        _log.skipped("txda not available — all dist patches skipped")
        return

    _patch_group_coordinator_device()
    _patch_all_gather_for_txda()

    _patched = True
