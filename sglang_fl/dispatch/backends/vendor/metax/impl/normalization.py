# MetaX normalization implementations.

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_metax(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    from sglang_fl.dispatch.backends.vendor.cuda.impl.normalization import (
        rms_norm_cuda,
    )

    return rms_norm_cuda(obj, x, residual)



def gemma_rms_norm_metax(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    from sglang_fl.dispatch.backends.vendor.cuda.impl.gemma_rms_norm import (
        gemma_rms_norm_cuda,
    )

    return gemma_rms_norm_cuda(obj, x, residual)
