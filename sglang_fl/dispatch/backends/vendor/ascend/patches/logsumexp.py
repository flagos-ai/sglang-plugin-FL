"""CANN 8.5 compatibility for SGLang's fused logprob top-k kernel.

On 910C, BiSheng IR compilation of ``_row_logsumexp_topk_kernel`` can
segfault when Triton Ascend enables automatic multi-buffering.  This path is
used by ``return_logprob`` with ``top_logprobs_num`` and is also SGLang's
speculative decode-vs-prefill conformance oracle.  Disable multi-buffering
only for this fused top-k kernel; the ordinary row-logsumexp kernel and all
other Triton launches retain their compiler defaults.
"""

from __future__ import annotations

from functools import wraps
import importlib
import logging
from typing import Any


logger = logging.getLogger(__name__)

_KERNEL_MODULE = "sglang.srt.layers.logsumexp"
_KERNEL_NAME = "_row_logsumexp_topk_kernel"
_PATCH_MARKER = "_sglang_fl_cann85_multibuffer_disabled"


def patch_logsumexp_topk_multibuffer() -> bool:
    """Force ``multibuffer=False`` for the fused logprob top-k kernel."""

    try:
        module = importlib.import_module(_KERNEL_MODULE)
        kernel = getattr(module, _KERNEL_NAME)
        original_run = kernel.run
    except (AttributeError, ImportError):
        logger.debug("SGLang fused logprob top-k kernel is unavailable", exc_info=True)
        return False

    if getattr(original_run, _PATCH_MARKER, False):
        return True

    @wraps(original_run)
    def run_without_multibuffer(*args: Any, **kwargs: Any) -> Any:
        kwargs["multibuffer"] = False
        return original_run(*args, **kwargs)

    setattr(run_without_multibuffer, _PATCH_MARKER, True)
    kernel.run = run_without_multibuffer
    logger.info(
        "Disabled Triton auto multi-buffering for %s.%s",
        _KERNEL_MODULE,
        _KERNEL_NAME,
    )
    return True
