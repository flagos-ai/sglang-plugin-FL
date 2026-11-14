"""FlagGems implementations of manifold-constrained hyper-connections."""

from __future__ import annotations

import torch

from sglang_fl.dispatch.logger_manager import log_exec


def _require_accelerator(tensor) -> None:
    if tensor.device.type == "cpu":
        raise RuntimeError("FlagGems mHC requires accelerator operands")


def _validate_mhc_pre_contract(
    hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value
) -> None:
    """Reject parameter combinations not represented by FlagGems kernels."""
    if float(hc_pre_eps) != float(hc_sinkhorn_eps):
        raise NotImplementedError(
            "FlagGems mHC requires hc_pre_eps == hc_sinkhorn_eps"
        )
    if float(hc_post_mult_value) != 2.0:
        raise NotImplementedError("FlagGems mHC requires hc_post_mult_value == 2.0")


def mhc_pre_flagos(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    """Run the portable FlagGems mHC decomposition."""

    _require_accelerator(residual)
    _validate_mhc_pre_contract(
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value
    )

    hc_mult = residual.shape[1]
    ori_dtype = residual.dtype
    flat = residual.float().flatten(1)
    reciprocal_rms = torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    mixes = torch.nn.functional.linear(flat, fn.float()) * reciprocal_rms

    from flag_gems.fused.mhc.hc_split_sinkhorn import hc_split_sinkhorn

    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_repeat,
        eps=hc_sinkhorn_eps,
    )
    layer_input = (residual * pre.to(ori_dtype).unsqueeze(-1)).sum(dim=1)
    result = (
        post.unsqueeze(-1),
        comb,
        layer_input.to(ori_dtype),
    )
    log_exec("mhc_pre", "default.flagos")
    return result


def mhc_post_flagos(x, residual, post_layer_mix, comb_res_mix):
    _require_accelerator(residual)

    from flag_gems.fused.mhc import mhc_post as flaggems_mhc_post

    result = flaggems_mhc_post(
        x, residual, post_layer_mix.float(), comb_res_mix.float()
    )
    log_exec("mhc_post", "default.flagos")
    return result
