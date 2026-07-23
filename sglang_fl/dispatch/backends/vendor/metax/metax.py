# MetaX/MACA backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class MetaxBackend(Backend):
    """MetaX backend for operator implementations.

    MetaX MACA exposes CUDA-compatible PyTorch APIs in current deployments.
    The per-op methods are routed through ``impl/`` wrappers so vendors can
    replace them with MACA-native kernels one by one.
    """

    _available: Optional[bool] = None
    _METAX_VENDOR_NAMES = {"metax", "maca"}

    @property
    def name(self) -> str:
        return "metax"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        if MetaxBackend._available is None:
            try:
                try:
                    from flag_gems.runtime.backend.device import DeviceDetector
                except ImportError:
                    from flag_gems.runtime.backend.device_finder import DeviceDetector
                vendor = DeviceDetector().vendor_name

                MetaxBackend._available = (
                    vendor in self._METAX_VENDOR_NAMES
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 0
                )
            except Exception:
                MetaxBackend._available = False
        return MetaxBackend._available

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_metax

        return silu_and_mul_metax(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_metax

        return rms_norm_metax(obj, x, residual)

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import gemma_rms_norm_metax

        return gemma_rms_norm_metax(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.rotary import rotary_embedding_metax

        return rotary_embedding_metax(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )


    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_metax

        return topk_metax(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_metax

        return fused_moe_metax(obj, layer, dispatch_output)

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
        from .impl.fla import chunk_gated_delta_rule_metax

        return chunk_gated_delta_rule_metax(
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
        from .impl.fla import fused_recurrent_gated_delta_rule_metax

        return fused_recurrent_gated_delta_rule_metax(
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
        from .impl.fla import fused_recurrent_gated_delta_rule_packed_decode_metax

        return fused_recurrent_gated_delta_rule_packed_decode_metax(
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
