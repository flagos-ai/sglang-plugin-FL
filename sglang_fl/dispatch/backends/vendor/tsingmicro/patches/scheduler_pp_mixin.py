# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch PP send/recv ordering for txda platform.

On txda, point-to-point ``isend`` is effectively blocking (same as XPU).
If every PP rank calls send first, all ranks block waiting for a
receiver and the ring deadlocks.  SGLang already orders send/recv by
``pp_rank`` parity for XPU inside
``SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors``::

    send_first = (not is_xpu()) or ((pp_rank % 2) == 0)

This patch runs SGLang's own method unchanged, except that ``is_xpu``
resolves to ``True`` inside that one function, so txda takes the XPU
ordering.  The method body is not copied: SGLang 0.5.18 changed it
(``self.pp_rank`` -> ``self.ps.pp_rank``, new ``_pp_can_skip_output_comm``
branch), and a copied 0.5.11 body crashes / desynchronizes PP on 0.5.18.
Works on 0.5.11 and 0.5.18.  The rebound function is built on the first
call, so it sees SGLang's module globals as late as possible.
"""

import types

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("scheduler_pp_mixin")

_METHOD = "_pp_send_recv_and_preprocess_output_tensors"
_patched = False
_originals = {}


def _is_xpu_for_txda() -> bool:
    return True


def _rebind_is_xpu(fn):
    """Copy of ``fn`` whose global ``is_xpu`` is ``_is_xpu_for_txda``.

    Nested functions created inside ``fn`` inherit the same globals, so the
    whole method body sees the override; nothing outside ``fn`` does.
    """
    if "is_xpu" not in fn.__code__.co_names:
        raise RuntimeError(f"{fn.__qualname__} no longer references is_xpu")
    new_globals = dict(fn.__globals__)
    new_globals["is_xpu"] = _is_xpu_for_txda
    new_fn = types.FunctionType(
        fn.__code__, new_globals, fn.__name__, fn.__defaults__, fn.__closure__
    )
    new_fn.__kwdefaults__ = fn.__kwdefaults__
    new_fn.__qualname__ = fn.__qualname__
    new_fn.__doc__ = fn.__doc__
    new_fn.__dict__.update(fn.__dict__)
    return new_fn


def patch() -> None:
    """Make txda use SGLang's XPU (even/odd) PP send/recv ordering.

    Idempotent: safe to call multiple times. No-op on non-txda platforms.
    """
    global _patched
    if _patched:
        return

    if not _is_txda():
        _log.skipped("txda not available - scheduler_pp_mixin patch skipped")
        return

    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        orig_fn = getattr(SchedulerPPMixin, _METHOD)
        if "is_xpu" not in orig_fn.__code__.co_names:
            _log.failed("%s does not reference is_xpu; ordering patch not applied", _METHOD)
            return
        _originals[_METHOD] = orig_fn
        cache = {}

        def _txda_pp_send_recv(self, *args, **kwargs):
            fn = cache.get("fn")
            if fn is None:
                fn = cache["fn"] = _rebind_is_xpu(orig_fn)
            return fn(self, *args, **kwargs)

        _txda_pp_send_recv.__qualname__ = orig_fn.__qualname__
        _txda_pp_send_recv.__doc__ = orig_fn.__doc__
        _txda_pp_send_recv.__wrapped__ = orig_fn
        setattr(SchedulerPPMixin, _METHOD, _txda_pp_send_recv)
        _patched = True
        _log.applied(
            "SchedulerPPMixin.%s -> upstream body with is_xpu()=True "
            "(txda uses even/odd PP send/recv ordering)", _METHOD
        )
    except Exception as exc:
        _log.failed("failed to patch %s: %s", _METHOD, exc)
