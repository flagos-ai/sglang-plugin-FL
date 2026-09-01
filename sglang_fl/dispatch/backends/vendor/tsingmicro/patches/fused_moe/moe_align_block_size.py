# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatching wrapper for ``moe_align_block_size`` that delegates to the
flag_gems pure-Triton implementation.

On txda (TsingMicro) hardware, the upstream ``sgl_kernel.moe_align_block_size``
cannot be imported because it depends on compiled CUDA kernels.  flag_gems
provides an equivalent pure-Triton implementation under
``flag_gems.runtime.backend._tsingmicro.fused.moe_align_block_size`` that
works on txda devices.

This module re-exports that implementation, adding optional parameters
(``expert_map``, ``pad_sorted_ids``) that the flag_gems kernel supports
but the sgl_kernel version does not.

See also:
  - :mod:`flag_gems.runtime.backend._tsingmicro.fused.moe_align_block_size`
  - :mod:`sglang_fl.dispatch.backends.vendor.tsingmicro.patches.fused_moe` — the patch that replaces the
    reference in sglang's fused_moe module with this implementation.
"""

from typing import Optional, Tuple

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("moe_align_block_size")

# ---------------------------------------------------------------------------
# Import flag_gems implementation — pure Triton, no CUDA dependency.
# ---------------------------------------------------------------------------
_IMPORT_ERROR: Optional[ImportError] = None
_flag_gems_moe_align_block_size = None

try:
    from flag_gems.runtime.backend._tsingmicro.fused.moe_align_block_size import (
        moe_align_block_size as _flag_gems_moe_align_block_size,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    _log.warning(
        "flag_gems moe_align_block_size import failed: %s. "
        "moe_align_block_size() will raise at call time.",
        exc,
    )


# ---------------------------------------------------------------------------
# Public API — mirrors sglang's moe_align_block_size signature, adds
# optional expert_map / pad_sorted_ids from flag_gems.
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align token distribution across experts for block matrix multiplication.

    Delegates to :func:`flag_gems.runtime.backend._tsingmicro.fused.\
moe_align_block_size.moe_align_block_size` — a pure-Triton kernel that
works on txda hardware without sgl_kernel.

    Parameters:
        topk_ids: ``[total_tokens, top_k]`` — top-k expert indices per token.
        block_size: The block size for block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: Optional tensor to remap expert IDs in the output.
        pad_sorted_ids: Whether to round up the sorted token count to a
            multiple of ``block_size`` after computing the padded count.

    Returns:
        - **sorted_token_ids** — padded sorted token indices ``[max_num_tokens_padded]``.
        - **expert_ids** — expert index per block ``[max_num_m_blocks]``.
        - **num_tokens_post_padded** — total token count after padding.
    """
    if _flag_gems_moe_align_block_size is None:
        raise ImportError(
            "flag_gems moe_align_block_size is not available. "
            "Is flag_gems installed with txda backend support?"
        ) from _IMPORT_ERROR

    return _flag_gems_moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        expert_map=expert_map,
        pad_sorted_ids=pad_sorted_ids,
    )
