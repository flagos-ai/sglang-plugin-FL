# TXDA (TsingMicro) backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class TxdaBackend(Backend):
    """
    TXDA backend for operator implementations.

    Uses TsingMicro torch_txda library.
    Ops that don't yet have a TXDA-native kernel raise NotImplementedError,
    which lets OpManager fall back to flaggems or reference automatically.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "txda"

    @property
    def vendor(self) -> Optional[str]:
        return "tsingmicro"

    def is_available(self) -> bool:
        """Check if TXDA hardware and torch_txda are available."""
        if TxdaBackend._available is None:
            try:
                TxdaBackend._available = (
                    hasattr(torch, "txda")
                    and torch.txda.is_available()
                    and torch.txda.device_count() > 0
                )
            except Exception:
                TxdaBackend._available = False
        return TxdaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_txda
        return silu_and_mul_txda(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_txda
        return rms_norm_txda(obj, x, residual)

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import gemma_rms_norm_txda
        return gemma_rms_norm_txda(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_txda
        return rotary_embedding_txda(
            obj, query, key, cos, sin, position_ids,
            rotary_interleaved=rotary_interleaved, inplace=inplace,
        )

    def mrotary_embedding(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.mrotary_embedding import mrotary_embedding_txda
        return mrotary_embedding_txda(obj, positions, query, key)

    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_txda
        return topk_txda(
            obj, hidden_states, router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_txda
        return fused_moe_txda(obj, layer, dispatch_output)

    def chunk_gated_delta_rule(
        self, q, k, v, g, beta, scale,
        initial_state=None, initial_state_indices=None,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import chunk_gated_delta_rule_txda
        return chunk_gated_delta_rule_txda(
            q, k, v, g, beta, scale,
            initial_state, initial_state_indices,
            cu_seqlens, head_first, use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule(
        self, q, k, v, g, beta, scale,
        initial_state=None, output_final_state=True,
        cu_seqlens=None, ssm_state_indices=None,
        num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import fused_recurrent_gated_delta_rule_txda
        return fused_recurrent_gated_delta_rule_txda(
            q, k, v, g, beta, scale,
            initial_state, output_final_state,
            cu_seqlens, ssm_state_indices,
            num_accepted_tokens, use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule_packed_decode(
        self, mixed_qkv, a, b, A_log, dt_bias, scale,
        initial_state, out, ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import fused_recurrent_gated_delta_rule_packed_decode_txda
        return fused_recurrent_gated_delta_rule_packed_decode_txda(
            mixed_qkv, a, b, A_log, dt_bias, scale,
            initial_state, out, ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
