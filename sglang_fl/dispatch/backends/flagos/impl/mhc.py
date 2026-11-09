# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FlagGems implementations of Xing4 mHC (multi-head composition) operators.

mHC collapses a residual stream of ``hc_mult`` copies into one hidden state
before the decoder layer and re-expands it afterwards. Xing4.0-29B-A4B uses
``hc_mult=4``; other widths are rejected explicitly rather than silently
producing a different result.

The sinkhorn split, the RMSNorm and the projection are dispatched to FlagGems
kernels so the pre/post pair stays on device without an fp32 round trip.
"""

from __future__ import annotations

import os
import weakref

import torch

from sglang_fl.dispatch.logger_manager import log_exec

# Casting the mHC projection weight to bf16 costs a full copy on every call.
# The weight is a frozen module parameter, so the cast is cached and dropped as
# soon as the source tensor dies.
_FN_BF16_CACHE: dict = {}

# npu_rms_norm needs a scale tensor; mHC applies an unweighted RMSNorm, so a ones
# vector is reused per (device, width, dtype) instead of reallocating each step.
_RMS_GAMMA_CACHE: dict = {}


def _get_bf16_weight(tensor: torch.Tensor) -> torch.Tensor:
    key = id(tensor)
    item = _FN_BF16_CACHE.get(key)
    if item is not None:
        ref, version, cached = item
        # Reuse only while the parameter is alive and has not been reassigned.
        if ref() is tensor and version == tensor._version:
            return cached

    cached = tensor.to(dtype=torch.bfloat16)

    def _drop(_ref, cache_key=key):
        _FN_BF16_CACHE.pop(cache_key, None)

    _FN_BF16_CACHE[key] = (weakref.ref(tensor, _drop), tensor._version, cached)
    return cached


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
    if residual.device.type != "npu":
        raise RuntimeError("FlagGems mHC requires NPU operands")

    n = residual.shape[1]
    if n != 4 or hc_post_mult_value != 2.0:
        raise NotImplementedError(
            "FlagGems mHC pre supports hc_mult=4 and hc_post_mult=2.0, "
            f"got hc_mult={n} and hc_post_mult={hc_post_mult_value}"
        )

    ori_dtype = residual.dtype

    if os.getenv("SGLANG_FL_MHC_PRE_FULL_FUSED", "0") == "1":
        from flag_gems.fused.mhc import mhc_pre as flaggems_mhc_pre

        post, comb, layer_input = flaggems_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        log_exec("mhc_pre", "default.flagos.full_fused")
        return (
            post.to(ori_dtype),
            comb.to(ori_dtype),
            layer_input.to(ori_dtype),
        )

    if os.getenv("SGLANG_FL_MHC_NPU_RMSNORM", "0") == "1":
        import torch_npu

        flat_bf16 = residual.flatten(1)
        gamma_key = (flat_bf16.device.index, flat_bf16.shape[-1], ori_dtype)
        gamma = _RMS_GAMMA_CACHE.get(gamma_key)
        if gamma is None:
            gamma = torch.ones(
                flat_bf16.shape[-1], device=flat_bf16.device, dtype=ori_dtype
            )
            _RMS_GAMMA_CACHE[gamma_key] = gamma
        normalized, _ = torch_npu.npu_rms_norm(flat_bf16, gamma, rms_eps)
    else:
        flat = residual.float().flatten(1)
        normalized = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + rms_eps
        )

    if os.getenv("SGLANG_FL_MHC_CACHE_FN_BF16", "0") == "1":
        fn_for_linear = _get_bf16_weight(fn)
    else:
        fn_for_linear = fn.to(ori_dtype)

    mixes = torch.nn.functional.linear(normalized.to(ori_dtype), fn_for_linear).float()

    from flag_gems.fused.mhc.hc_split_sinkhorn import hc_split_sinkhorn

    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=n,
        sinkhorn_iters=sinkhorn_repeat,
        eps=hc_sinkhorn_eps,
    )
    layer_input = (residual * pre.to(ori_dtype).unsqueeze(-1)).sum(dim=1)
    log_exec("mhc_pre", "default.flagos")
    return (
        post.to(ori_dtype).unsqueeze(-1),
        comb.to(ori_dtype),
        layer_input.to(ori_dtype),
    )


def mhc_post_flagos(x, residual, post_layer_mix, comb_res_mix):
    if residual.device.type != "npu":
        raise RuntimeError("FlagGems mHC requires NPU operands")

    # The FlagGems mhc_post kernel accumulates in fp32 but stores bf16
    # unconditionally (out = torch.empty_like(residual), tl.store(..., .to(bf16))).
    # Accepting an fp32 residual here would silently return bf16-precision values,
    # so reject it and let the dispatch layer pick a backend that honours the
    # requested dtype.
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError(
            "FlagGems mHC post only supports bfloat16 residual streams, "
            f"got residual.dtype={residual.dtype}"
        )

    from flag_gems.fused.mhc import mhc_post as flaggems_mhc_post

    result = flaggems_mhc_post(
        x, residual, post_layer_mix.float(), comb_res_mix.float()
    )
    log_exec("mhc_post", "default.flagos")
    return result
