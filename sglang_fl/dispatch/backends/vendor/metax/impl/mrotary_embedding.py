# MetaX multimodal rotary embedding implementations.

from __future__ import annotations

import torch


def mrotary_embedding_metax(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang_fl.dispatch.backends.vendor.cuda.impl.mrotary_embedding import (
        mrotary_embedding_cuda,
    )

    return mrotary_embedding_cuda(obj, positions, query, key)
