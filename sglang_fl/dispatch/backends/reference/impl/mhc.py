"""PyTorch reference implementations for manifold-constrained hyper-connections."""

from __future__ import annotations

import torch

from sglang_fl.dispatch.logger_manager import log_exec


def mhc_pre_torch(
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
    hc_mult = residual.shape[1]
    ori_dtype = residual.dtype
    flat = residual.float().flatten(1)
    reciprocal_rms = torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    mixes = torch.nn.functional.linear(flat, fn.float()) * reciprocal_rms

    pre = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
        + hc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb = (
        mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    ).reshape(-1, hc_mult, hc_mult)
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (residual.float() * pre.unsqueeze(-1)).sum(dim=1)
    result = (
        post.unsqueeze(-1),
        comb,
        layer_input.to(ori_dtype),
    )
    log_exec("mhc_pre", "reference.torch")
    return result


def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    result = (
        x.float().unsqueeze(1) * post_layer_mix.float()
        + torch.bmm(comb_res_mix.float().mT, residual.float())
    ).to(x.dtype)
    log_exec("mhc_post", "reference.torch")
    return result
