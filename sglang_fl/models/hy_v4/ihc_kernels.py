# SPDX-License-Identifier: Apache-2.0
"""Optional HPC iHC delegation; device kernels belong to operator libraries."""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)
_HPC_IHC_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
_HPC_IHC_HC_MULTS = (4,)
_HPC_IHC_HIDDEN_SIZES = (4096, 6144)


@functools.cache
def _hpc_ihc_op(op_name: str, hc_mult: int, hidden_size: int):
    try:
        import hpc
    except ModuleNotFoundError as error:
        if error.name != "hpc":
            raise
        return None
    op = getattr(hpc, op_name, None)
    if op is None:
        logger.info(
            "HY4 iHC: the installed hpc build (%s) has no %s; using the model Torch fallback.",
            getattr(hpc, "__version__", "unknown"),
            op_name,
        )
        return None
    from sglang.srt.utils import get_device_capability

    cap = get_device_capability()
    if cap not in _HPC_IHC_CAPABILITIES:
        logger.warning("HY4 iHC: hpc.%s is unavailable on sm%s%s.", op_name, *cap)
        return None
    if hc_mult not in _HPC_IHC_HC_MULTS or hidden_size not in _HPC_IHC_HIDDEN_SIZES:
        logger.warning(
            "HY4 iHC: hpc.%s is instantiated for hc_mult in %s and hidden_size in %s, got %d / %d.",
            op_name,
            _HPC_IHC_HC_MULTS,
            _HPC_IHC_HIDDEN_SIZES,
            hc_mult,
            hidden_size,
        )
        return None
    logger.info("HY4 iHC: using hpc.%s.", op_name)
    return op


def fused_hy4_ihc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    hc_mult = x.shape[1]
    hidden_size = x.shape[-1]
    hpc_op = _hpc_ihc_op("fuse_ihc_pre", hc_mult, hidden_size)
    if hpc_op is None:
        raise NotImplementedError(
            "optional HPC iHC API is unavailable; use model fallback"
        )
    return hpc_op(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
        True,
    )


def fused_hy4_ihc_post(
    output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    hc_mult = post.shape[-1]
    hidden_size = output.shape[-1]
    hpc_op = _hpc_ihc_op("fuse_ihc_post", hc_mult, hidden_size)
    if hpc_op is None:
        raise NotImplementedError(
            "optional HPC iHC API is unavailable; use model fallback"
        )
    return hpc_op(output, residual, post)


def fused_hy4_ihc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> torch.Tensor:
    hc_mult = hidden_states.shape[1]
    hidden_size = hidden_states.shape[-1]
    hpc_op = _hpc_ihc_op("fuse_ihc_head", hc_mult, hidden_size)
    if hpc_op is None:
        raise NotImplementedError(
            "optional HPC iHC API is unavailable; use model fallback"
        )
    return hpc_op(
        hidden_states.contiguous(),
        hc_fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        norm_eps,
        hc_eps,
        rms_weight,
        rms_eps,
        True,
    )
