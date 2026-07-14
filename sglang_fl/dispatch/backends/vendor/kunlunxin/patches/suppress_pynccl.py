"""Kunlunxin: suppress the real ctypes PyNccl in GroupCoordinator under FlagCX.

On Kunlun the device is CUDA-compat (torch reports ``cuda``, not ``xpu``), so
sglang's ``init_model_parallel_group`` computes
``use_pynccl = not (_is_npu or _is_xpu or backend == "mooncake")`` == ``True``
and ``GroupCoordinator.__init__`` eagerly builds a real ctypes
``PyNcclCommunicator`` whose ``ncclCommInitRank`` aborts with
``NCCL error: unhandled cuda error`` on Kunlun.

When the FL distributed backend is FlagCX, ``CommunicatorFL`` drives every
collective, so PyNccl is unnecessary. This patch wraps
``GroupCoordinator.__init__`` and forces ``use_pynccl=False`` *before* the real
``__init__`` runs, preventing the crash. ``GroupCoordinator`` is always called
with ``use_pynccl`` as a keyword (``init_model_parallel_group``), so the kwargs
path suffices.

Load ordering: this vendor patch is applied (setattr) during
``_apply_vendor_patches`` — after the plugin's AROUND ``__init__`` hook is
*registered* but before ``HookRegistry.apply_hooks`` captures the current
``GroupCoordinator.__init__`` as its ``original_fn``. The AROUND wrapper
therefore wraps *this* patched ``__init__``, so the suppression runs just before
the real constructor.

Only active when the FL backend is FlagCX (``SGLANG_FL_DIST_BACKEND=flagcx`` or
``FLAGCX_PATH`` set); otherwise sglang is left untouched. No-op at
``world_size==1`` (``use_pynccl`` only builds a communicator when
``world_size>1``), so single-card runs are unaffected either way.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_suppress_pynccl():
    """Force ``use_pynccl=False`` in GroupCoordinator.__init__ when FlagCX is active."""
    global _applied
    if _applied:
        return

    from sglang.srt.distributed import parallel_state as ps

    orig_init = ps.GroupCoordinator.__init__
    if getattr(orig_init, "_klx_pynccl_suppressed", False):
        _applied = True
        return

    def _init(self, *args, **kwargs):
        if kwargs.get("use_pynccl"):
            kwargs["use_pynccl"] = False
        return orig_init(self, *args, **kwargs)

    _init._klx_pynccl_suppressed = True
    ps.GroupCoordinator.__init__ = _init

    _applied = True
    logger.info(
        "patched GroupCoordinator.__init__ -> use_pynccl=False on Kunlunxin "
        "(FlagCX drives collectives; avoids real PyNccl ncclCommInitRank crash)"
    )
