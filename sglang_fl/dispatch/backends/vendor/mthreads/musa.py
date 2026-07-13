# MUSA (Moore Threads) backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class MusaBackend(Backend):
    """
    MUSA backend for operator implementations.

    Uses Moore Threads MUSA libraries (torch_musa).
    Ops that don't yet have a MUSA-native kernel raise NotImplementedError,
    which lets OpManager fall back to flaggems or reference automatically.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "musa"

    @property
    def vendor(self) -> Optional[str]:
        return "mthreads"

    def is_available(self) -> bool:
        """Check if MUSA hardware and torch_musa are available."""
        if MusaBackend._available is None:
            try:
                MusaBackend._available = (
                    hasattr(torch, "musa")
                    and torch.musa.is_available()
                    and torch.musa.device_count() > 0
                )
            except Exception:
                MusaBackend._available = False
        return MusaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_musa

        return silu_and_mul_musa(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_musa

        return rms_norm_musa(obj, x, residual)

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import gemma_rms_norm_musa

        return gemma_rms_norm_musa(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_musa

        return rotary_embedding_musa(
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
        from .impl.mrotary_embedding import mrotary_embedding_musa

        return mrotary_embedding_musa(obj, positions, query, key)

    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_musa

        return topk_musa(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_musa

        return fused_moe_musa(obj, layer, dispatch_output)

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
        from .impl.fla import chunk_gated_delta_rule_musa

        return chunk_gated_delta_rule_musa(
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
        from .impl.fla import fused_recurrent_gated_delta_rule_musa

        return fused_recurrent_gated_delta_rule_musa(
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
        from .impl.fla import fused_recurrent_gated_delta_rule_packed_decode_musa

        return fused_recurrent_gated_delta_rule_packed_decode_musa(
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
