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
"""PyTorch reference implementation of the Xing4 mHC operators.

This is the definition the FlagGems backend is validated against: it follows the
arithmetic of the Xing4 checkpoint's own modeling file step by step and uses
only eager PyTorch, so it shares no kernel with the implementation under test.
"""

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
    n = residual.shape[1]
    ori_dtype = residual.dtype

    flat = residual.float().flatten(1)
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)

    mixes = torch.nn.functional.linear(
        normalized.to(ori_dtype), fn.to(ori_dtype)
    ).float()

    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n])
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
        * hc_post_mult_value
    )

    comb = (mixes[:, 2 * n :] * hc_scale[2] + hc_base[2 * n :]).reshape(-1, n, n)
    comb = torch.clamp(comb, min=-30.0, max=30.0)
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    for _ in range(sinkhorn_repeat):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (residual * pre.to(ori_dtype).unsqueeze(-1)).sum(dim=1)
    log_exec("mhc_pre", "reference.torch")
    return (
        post.to(ori_dtype).unsqueeze(-1),
        comb.to(ori_dtype),
        layer_input.to(ori_dtype),
    )


def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    out = post_layer_mix * x.float().unsqueeze(1)
    for source in range(residual.shape[1]):
        out = out + (
            comb_res_mix[:, source, :].unsqueeze(-1)
            * residual[:, source, :].float().unsqueeze(1)
        )
    log_exec("mhc_post", "reference.torch")
    return out.to(residual.dtype)
