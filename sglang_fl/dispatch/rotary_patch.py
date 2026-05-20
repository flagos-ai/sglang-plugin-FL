# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch RotaryEmbedding / MRotaryEmbedding __init__ to restore dispatch bridge
after MUSA override.

Problem
-------
On MUSA, ``RotaryEmbedding.__init__`` (base.py:103-107) unconditionally overwrites
``self._forward_method = self.forward_native`` *after* ``MultiPlatformOp.__init__`` has
already stored the bridge returned by ``dispatch_forward()`` (which the plugin's AROUND
hook intercepts).  The stomp happens for every ``RotaryEmbedding`` instance, including
``MRotaryEmbedding`` subclasses, because ``MRotaryEmbedding.__init__`` delegates to
``super().__init__()`` → ``RotaryEmbedding.__init__``.

Fix strategy
------------
Two separate wrappers, each using an **exact-type guard** (``type(self) is X``) so that
the re-dispatch only fires for the class whose ``__init__`` was patched, never for an
unknown subclass that may still be mid-construction:

1. ``_patched_rope_init`` wraps ``RotaryEmbedding.__init__``.
   Only re-dispatches when ``type(self) is RotaryEmbedding``.  Subclasses whose
   ``super().__init__()`` passes through here are left untouched; each subclass that also
   needs the fix gets its own patched ``__init__`` (see point 2).

2. ``_patched_mrope_init`` wraps ``MRotaryEmbedding.__init__``.
   Only re-dispatches when ``type(self) is MRotaryEmbedding``, after ``mrope_section``,
   ``mrope_interleaved`` etc. are fully set — no partial-init window.

Both wrappers are gated on ``is_musa()`` and are no-ops on other platforms.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_rotary_embedding_init() -> None:
    """Patch ``RotaryEmbedding`` and ``MRotaryEmbedding`` ``__init__`` on MUSA.

    Safe to call multiple times (idempotent).  Must be called *after* the AROUND hook on
    ``MultiPlatformOp.dispatch_forward`` has been registered so that the re-invocation of
    ``dispatch_forward()`` returns the bridge instead of ``forward_native``.
    """
    global _patched
    if _patched:
        return

    try:
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
        from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
        from sglang.srt.utils import is_musa

        if not is_musa():
            logger.debug(
                "rotary_patch: not on MUSA – RotaryEmbedding init patch skipped"
            )
            return

        _orig_rope_init = RotaryEmbedding.__init__
        _orig_mrope_init = MRotaryEmbedding.__init__

        def _patched_rope_init(self, *args, **kwargs):
            _orig_rope_init(self, *args, **kwargs)
            # Exact-type guard: only re-dispatch for RotaryEmbedding itself.
            # Subclasses (MRotaryEmbedding, YaRNScaling*, Llama3*, …) each have their own
            # __init__ and their own fields that may not be set yet when super().__init__()
            # returns here.  Each subclass that needs the fix gets its own patched __init__.
            if type(self) is not RotaryEmbedding:
                return
            self._forward_method = self.dispatch_forward()

        def _patched_mrope_init(self, *args, **kwargs):
            _orig_mrope_init(self, *args, **kwargs)
            # Exact-type guard: only re-dispatch for MRotaryEmbedding itself.
            # After _orig_mrope_init returns, mrope_section / mrope_interleaved / … are
            # all fully set, so there is no partial-init window.
            if type(self) is not MRotaryEmbedding:
                return
            self._forward_method = self.dispatch_forward()

        RotaryEmbedding.__init__ = _patched_rope_init
        MRotaryEmbedding.__init__ = _patched_mrope_init

        _patched = True
        logger.info(
            "rotary_patch: RotaryEmbedding + MRotaryEmbedding __init__ patched – "
            "rotary_embedding / mrotary_embedding bridges active on MUSA"
        )

    except Exception as exc:
        logger.error(
            "rotary_patch: failed to patch rotary embedding __init__: %s", exc
        )
