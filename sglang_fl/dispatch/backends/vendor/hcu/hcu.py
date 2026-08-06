# Template backend implementation.
#
# Copy this directory as your starting point for a new vendor backend.
# Replace 'template' with your vendor name throughout.

from __future__ import annotations

from typing import Optional, Union

from sglang_fl.dispatch.backends import Backend

import torch


class hcuBackend(Backend):
    """
    Template backend for operator implementations.

    Replace this with your vendor-specific backend.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "hcu"

    @property
    def vendor(self) -> Optional[str]:
        return "hcu"

    def is_available(self) -> bool:
        """Check if HCU hardware and runtime are available."""
        if hcuBackend._available is None:
            try:
                hcuBackend._available = (
                    hasattr(torch, "__hcu_version__")
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 0
                )
            except Exception:
                hcuBackend._available = False
        return hcuBackend._available

    # ==================== Operator Implementations ====================
    def chunk_gated_delta_rule(self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float = None,
            initial_state: torch.Tensor = None,
            initial_state_indices: torch.Tensor = None,
            cu_seqlens: Optional[torch.LongTensor] = None,
            head_first: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
    ):
        from .impl.fla_chunk import chunk_gated_delta_rule_hcu
        return chunk_gated_delta_rule_hcu(
            q, k, v, g, beta, scale, initial_state, initial_state_indices,
            cu_seqlens, head_first, use_qk_l2norm_in_kernel
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
        from .impl.fla_fused_recurrent import fused_recurrent_gated_delta_rule_hcu

        return fused_recurrent_gated_delta_rule_hcu(
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
    def fused_recurrent_gated_delta_rule_packed_decode(self,
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
    ):
        from .impl.fla_packed_decode import fused_recurrent_gated_delta_rule_packed_decode_hcu
        return fused_recurrent_gated_delta_rule_packed_decode_hcu(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, ssm_state_indices, use_qk_l2norm_in_kernel)
    

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_hcu

        return fused_moe_hcu(obj, layer, dispatch_output)
    
    def gemma_rms_norm(self, obj,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
    ):
        from .impl.gemma_rms_norm import gemma_rms_norm_hcu
        return gemma_rms_norm_hcu(obj, x, residual)

    def mrotary_embedding(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.mrotary_embedding import mrotary_embedding_hcu

        return mrotary_embedding_hcu(obj, positions, query, key)
    
    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        from .impl.rms_norm import rms_norm_hcu

        return rms_norm_hcu(obj, x, residual)
    
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
        from .impl.rotary_embedding import rotary_embedding_hcu

        return rotary_embedding_hcu(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )
    
    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.silu_and_mul import silu_and_mul_hcu
        return silu_and_mul_hcu(obj, x)
    
    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_hcu

        return topk_hcu(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


    # def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
    #     from .impl.activation import silu_and_mul_template
    #     return silu_and_mul_template(obj, x)
    #
    # def rms_norm(self, obj, x, residual=None):
    #     from .impl.normalization import rms_norm_template
    #     return rms_norm_template(obj, x, residual)
    #
    # def rotary_embedding(self, obj, query, key, cos, sin, position_ids,
    #                      rotary_interleaved=False, inplace=True):
    #     from .impl.rotary import rotary_embedding_template
    #     return rotary_embedding_template(
    #         obj, query, key, cos, sin, position_ids,
    #         rotary_interleaved, inplace,
    #     )
