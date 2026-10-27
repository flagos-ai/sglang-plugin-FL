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

# AMD ROCm backend.
#
# Covers the ops that have no FlagOS implementation yet. They reuse SGLang's
# own Triton kernels (same as the CUDA backend), which compile on ROCm Triton;
# sgl_kernel / aiter are not used (see patch.py for the fused-MoE helpers).

from __future__ import annotations

import torch

from sglang_fl.dispatch.backends import Backend


class AmdBackend(Backend):
    _available: bool | None = None

    @property
    def name(self) -> str:
        return "amd"

    @property
    def vendor(self) -> str | None:
        return "amd"

    def is_available(self) -> bool:
        if AmdBackend._available is None:
            try:
                from sglang.srt.platforms import current_platform

                AmdBackend._available = (
                    torch.version.hip is not None
                    and torch.cuda.is_available()
                    and getattr(current_platform, "_vendor_name", None) == "amd"
                )
            except (ImportError, RuntimeError):
                AmdBackend._available = False
        return AmdBackend._available

    # ==================== MoE ====================

    def fused_moe(self, obj, layer, dispatch_output):
        from sglang_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            fused_moe_cuda,
        )

        return fused_moe_cuda(obj, layer, dispatch_output)

    # ==================== FLA (Gated DeltaNet) ====================

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        initial_state_indices=None,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    ):
        from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
            chunk_gated_delta_rule_cuda,
        )

        return chunk_gated_delta_rule_cuda(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            initial_state_indices,
            cu_seqlens,
            head_first,
            use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    ):
        from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
            fused_recurrent_gated_delta_rule_cuda,
        )

        return fused_recurrent_gated_delta_rule_cuda(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule_packed_decode(
        self,
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        from sglang_fl.dispatch.backends.vendor.cuda.impl.fla import (
            fused_recurrent_gated_delta_rule_packed_decode_cuda,
        )

        return fused_recurrent_gated_delta_rule_packed_decode_cuda(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
