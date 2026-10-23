# MetaX rotary embedding implementations.

from __future__ import annotations

import torch


def rotary_embedding_metax(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang_fl.dispatch.backends.vendor.cuda.impl.rotary import (
        rotary_embedding_cuda,
    )

    return rotary_embedding_cuda(
        obj,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )
