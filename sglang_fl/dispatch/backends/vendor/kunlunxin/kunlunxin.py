# Kunlunxin (Baidu Kunlun XPU) backend implementation.
#
# Operators are implemented entirely on top of the ``sgl_kernel`` package (Kunlun
# klx-backed ``torch.ops.sgl_kernel.*`` ops), including the Gated-DeltaNet paths
# which route through ``torch.ops.sgl_kernel.klx_gated_delta_net``.

from __future__ import annotations

from typing import Optional

import torch

from sglang_fl.dispatch.backends import Backend


class KunlunxinBackend(Backend):
    """Kunlunxin Platform operator backend."""

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "kunlunxin"

    @property
    def vendor(self) -> Optional[str]:
        return "kunlunxin"

    def is_available(self) -> bool:
        if KunlunxinBackend._available is None:
            try:
                import torch_xmlir

                KunlunxinBackend._available = (
                    torch_xmlir.xpu.is_available()
                    and torch_xmlir.get_xpu_version() == "KL3"
                )
            except Exception:
                KunlunxinBackend._available = False
        return KunlunxinBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_kunlunxin

        return silu_and_mul_kunlunxin(obj, x)

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
        from .impl.fla import chunk_gated_delta_rule_kunlunxin

        return chunk_gated_delta_rule_kunlunxin(
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
        from .impl.fla import (
            fused_recurrent_gated_delta_rule_packed_decode_kunlunxin,
        )

        return fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
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

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_kunlunxin

        return fused_moe_kunlunxin(obj, layer, dispatch_output)
