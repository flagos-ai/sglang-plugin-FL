"""CANN 8.5 compatibility for SGLang's fused logprob top-k kernel.

On 910C, BiSheng IR cannot compile ``_row_logsumexp_topk_kernel``.  The
failure remains when automatic multi-buffering is disabled, so changing a
compiler option is not a sufficient workaround.

Replace only the public fused helper with a split implementation: SGLang's
ordinary single-pass ``row_logsumexp`` kernel still computes the fp32 row
normalizer, while ``torch.topk`` selects the raw logits.  This retains the
low-memory normalizer and avoids materializing a full-vocabulary log-softmax
tensor.  It also matches SGLang's existing non-fused top-k behavior.
"""

from __future__ import annotations

import importlib
import logging
from functools import wraps

import torch


logger = logging.getLogger(__name__)

_KERNEL_MODULE = "sglang.srt.layers.logsumexp"
_TOPK_NAME = "row_logsumexp_topk"
_ROW_LSE_NAME = "row_logsumexp"
_FUSED_LIMIT_NAME = "FUSED_TOPK_MAX_K"
_PATCH_MARKER = "_sglang_fl_ascend_split_topk"


def patch_logsumexp_topk_fallback() -> bool:
    """Replace the fused helper with row-logsumexp plus ``torch.topk``."""

    try:
        module = importlib.import_module(_KERNEL_MODULE)
        original_topk = getattr(module, _TOPK_NAME)
        row_logsumexp = getattr(module, _ROW_LSE_NAME)
        fused_max_k = getattr(module, _FUSED_LIMIT_NAME)
    except (AttributeError, ImportError):
        logger.debug("SGLang logprob top-k helpers are unavailable", exc_info=True)
        return False

    if getattr(original_topk, _PATCH_MARKER, False):
        return True

    @wraps(original_topk)
    def split_row_logsumexp_topk(
        x: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 2
        assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        num_cols = x.shape[1]
        assert 1 <= k <= min(fused_max_k, num_cols), (k, num_cols)

        row_max, row_log_sum = row_logsumexp(x)
        top_vals, top_idx = torch.topk(
            x,
            k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        return row_max, row_log_sum, top_vals.float(), top_idx

    setattr(split_row_logsumexp_topk, _PATCH_MARKER, True)
    setattr(module, _TOPK_NAME, split_row_logsumexp_topk)
    logger.info(
        "Replaced %s.%s with row-logsumexp plus PyTorch top-k",
        _KERNEL_MODULE,
        _TOPK_NAME,
    )
    return True
