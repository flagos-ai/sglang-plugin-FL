# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional flag_gems ``_tsingmicro`` ops, resolved without hard imports.

The TXDA backend accelerates three ops with kernels that live in flag_gems'
``_tsingmicro`` fused backend::

    extend_attention_fwd          (prefill / extend attention)
    decode_attention_fwd_sglang   (decode attention, SGLang CSR layout)
    chunk_gated_delta_rule        (Gated DeltaNet prefill)

They are relatively recent additions.  An older flag_gems release ships the
``_tsingmicro`` backend *without* them (or without the backend entirely), and
importing them unconditionally would take the whole plugin down at load time.

:func:`load_txda_op` resolves one of them lazily and returns ``None`` instead of
raising, so every call site can fall back to SGLang's own kernel.  Results are
cached, so the "not available" warning is emitted at most once per process.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TXDA_FUSED = "flag_gems.runtime.backend._tsingmicro.fused"

# Plugin-facing op name -> (flag_gems module, attribute name).
_TXDA_OPS = {
    "extend_attention_fwd": (
        f"{_TXDA_FUSED}.extend_attention",
        "extend_attention_fwd",
    ),
    "decode_attention_fwd_sglang": (
        f"{_TXDA_FUSED}.decode_attention",
        "decode_attention_fwd_sglang",
    ),
    "chunk_gated_delta_rule": (
        f"{_TXDA_FUSED}.fla",
        "chunk_gated_delta_rule",
    ),
}

_cache: dict[str, Optional[Callable]] = {}


def load_txda_op(op_name: str) -> Optional[Callable]:
    """Return the flag_gems ``_tsingmicro`` implementation of ``op_name``.

    Returns ``None`` when flag_gems is not installed, the ``_tsingmicro`` backend
    is absent, or this particular op is missing from it -- i.e. on any flag_gems
    release predating the op.  Callers must fall back to SGLang's kernel then.
    """
    if op_name in _cache:
        return _cache[op_name]

    module_path, attr = _TXDA_OPS[op_name]
    fn: Optional[Callable] = None
    try:
        module = importlib.import_module(module_path)
        fn = getattr(module, attr, None)
    except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
        logger.warning(
            "flag_gems op '%s' unavailable (%s: %s); falling back to SGLang's kernel",
            op_name,
            type(exc).__name__,
            exc,
        )
    else:
        if fn is None:
            logger.warning(
                "flag_gems op '%s' not found in %s; falling back to SGLang's kernel",
                op_name,
                module_path,
            )

    _cache[op_name] = fn
    return fn
