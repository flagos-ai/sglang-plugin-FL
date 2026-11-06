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

# MUSA FLA (Flash Linear Attention) operator implementations.
#
# TODO: Replace SGLang's original triton kernels with torch_musa native kernel once verified
# on hardware. Current behavior: SGLang's original triton kernels.

from __future__ import annotations

import inspect
import os
from typing import Optional, Tuple

import torch


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn


def chunk_gated_delta_rule_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    inplace_update: bool = True,
):
    """Use FlagGems for the packed-varlen state update when enabled."""
    if (
        os.environ.get("SGLANG_MUSA_FLAGOS_CHUNK_GDN", "0") == "1"
        and initial_state is not None
        and initial_state_indices is not None
        and cu_seqlens is not None
        and not head_first
        and inplace_update
    ):
        from flag_gems.fused.chunk_gated_delta_rule import (
            chunk_gated_delta_rule as flagos_chunk_gated_delta_rule,
        )

        # SGLang uses [N, H, V, K]; FlagGems consumes [N, H, K, V].
        # K and V can be equal, so shape equality does not prove layout.
        selected_state = (
            initial_state[initial_state_indices].transpose(-1, -2).contiguous()
        )
        output, final_state = flagos_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta.to(q.dtype),
            g=g.to(q.dtype),
            initial_state=selected_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens.to(torch.long),
            head_first=False,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        initial_state[initial_state_indices] = final_state.transpose(-1, -2).to(
            initial_state.dtype
        )
        return output, None, None

    original = _original("chunk_gated_delta_rule")
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    parameters = inspect.signature(original).parameters
    if "inplace_update" in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        kwargs["inplace_update"] = inplace_update
    elif not inplace_update:
        raise NotImplementedError(
            "This SGLang FLA version does not support inplace_update=False"
        )
    return original(**kwargs)


def fused_recurrent_gated_delta_rule_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """fused_recurrent_gated_delta_rule — not yet implemented on MUSA. Current behavior: SGLang's original triton kernels."""
    return _original("fused_recurrent_gated_delta_rule")(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_musa(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fused_recurrent_gated_delta_rule_packed_decode — not yet implemented on MUSA. Current behavior: SGLang's original triton kernels."""
    return _original("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
