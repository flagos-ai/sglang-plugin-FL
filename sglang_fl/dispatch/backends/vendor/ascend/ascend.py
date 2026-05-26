# Ascend backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    Uses Ascend CANN libraries for Huawei Ascend NPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> Optional[str]:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            try:
                if torch.npu.is_available() and torch.npu.device_count() > 0:
                    AscendBackend._available = True
                else:
                    AscendBackend._available = False
            except Exception:
                AscendBackend._available = False
        return AscendBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_ascend

        return silu_and_mul_ascend(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_ascend

        return rms_norm_ascend(obj, x, residual)

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import gemma_rms_norm_ascend

        return gemma_rms_norm_ascend(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_ascend

        return rotary_embedding_ascend(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def mrotary_embedding(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.mrotary_embedding import mrotary_embedding_ascend

        return mrotary_embedding_ascend(obj, positions, query, key)

    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_ascend

        return topk_ascend(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_ascend

        return fused_moe_ascend(obj, layer, dispatch_output)

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
        from .impl.fla import chunk_gated_delta_rule_ascend

        return chunk_gated_delta_rule_ascend(
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
