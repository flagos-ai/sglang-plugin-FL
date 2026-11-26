# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disable SGLang's CUDA-only Triton load watch on txda (SGLang >= 0.5.18).

SGLang 0.5.18 added ``sglang.srt.utils.triton_load_watch``.  The scheduler
calls ``install()`` + ``mark_serving_started()`` once the engine is up; after
that, every first-use Triton kernel load runs ``_on_kernel_load``, which calls
``get_available_gpu_memory("cuda", ...)``.  On txda ``torch.cuda`` is mapped to
torch_txda, and ``_TxdaDeviceProperties`` has no ``is_integrated`` attribute,
so the hook raises ``AttributeError`` (only ``RuntimeError`` is caught) inside
the FlagGems kernel launch and kills every TP rank at the first request.

The watch only emits OOM diagnostics tied to CUDA ``cuModuleLoadData``; it has
no effect on results.  Make ``install()`` a no-op on txda.  The scheduler calls
``triton_load_watch.install()`` through the module attribute, so rebinding the
attribute is enough.  Older SGLang without the module: skipped.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("triton_load_watch")

_patched = False


def _noop_install() -> None:
    return None


def patch() -> None:
    """Make triton_load_watch.install a no-op. Idempotent."""
    global _patched
    if _patched:
        return
    try:
        from sglang.srt.utils import triton_load_watch as tlw
    except ImportError:
        _log.skipped("sglang.srt.utils.triton_load_watch not present (SGLang < 0.5.18)")
        return
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot import triton_load_watch: %s", exc)
        return

    if getattr(tlw, "_installed", False):
        _log.warning("triton_load_watch already installed before patch; hook stays active")
    tlw.install = _noop_install
    _patched = True
    _log.applied("triton_load_watch.install disabled (CUDA-only kernel-load diagnostics)")
