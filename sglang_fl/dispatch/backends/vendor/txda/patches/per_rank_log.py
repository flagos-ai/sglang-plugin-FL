# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redirect scheduler subprocess stdout/stderr to per-rank log files.

Controlled by ``SGLANG_FL_LOG_DIR`` environment variable.  When set, each
scheduler subprocess writes its output to ``{SGLANG_FL_LOG_DIR}/rank_TP{n}.log``
instead of inheriting the parent's stdout/stderr, keeping multi-rank logs
cleanly separated.

Uses SGLang's HookRegistry (string-path registration, lazy resolution) to
avoid importing ``sglang.srt.managers.scheduler`` directly.  Direct import
of that module triggers the full sglang import chain, which accesses
``torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`` — unavailable
after ``torch_txda`` has remapped ``torch.cuda`` → ``torch.txda``.

The hook intercepts ``configure_scheduler_process`` before SGLang's
``configure_logger`` creates its ``StreamHandler``, so the handler points to
the redirected stream rather than the original console.
"""

import os
import sys

from sglang_fl.dispatch.backends.vendor.txda.patches._logger import patch_logger

_log = patch_logger("per_rank_log")

_patched = False


def patch() -> None:
    """Apply per-rank log redirection if ``SGLANG_FL_LOG_DIR`` is set.

    Idempotent: safe to call multiple times.
    """
    global _patched
    if _patched:
        return

    log_dir = os.environ.get("SGLANG_FL_LOG_DIR", "").strip()
    if not log_dir:
        _log.skipped("SGLANG_FL_LOG_DIR not set — per-rank log redirection skipped")
        _patched = True
        return

    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        def _per_rank_log_hook(original_fn, *args, **kwargs):
            tp_rank = args[2] if len(args) > 2 else kwargs.get("tp_rank", "unknown")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"rank_TP{tp_rank}.log")
            _log_f = open(log_path, "a", buffering=1)
            sys.stdout = _log_f
            sys.stderr = _log_f
            return original_fn(*args, **kwargs)

        HookRegistry.register(
            "sglang.srt.managers.scheduler.configure_scheduler_process",
            _per_rank_log_hook,
            HookType.AROUND,
        )

        _patched = True
        _log.applied(
            "per-rank log redirection enabled → %s/rank_TP{n}.log", log_dir
        )
    except Exception as exc:
        _log.failed("failed to patch per-rank log redirection: %s", exc)
