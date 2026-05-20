# CUDA backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class CudaBackend(Backend):
    """
    CUDA backend for operator implementations.

    Uses sgl_kernel (SGLang native CUDA kernels) for NVIDIA GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> Optional[str]:
        return "nvidia"

    def is_available(self) -> bool:
        """Check if CUDA hardware and sgl_kernel are available."""
        if CudaBackend._available is None:
            try:
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    CudaBackend._available = False
                    return False
                # Use platform's vendor_name from FlagGems DeviceDetector
                # to distinguish real NVIDIA GPUs from CUDA-alike devices
                # (Iluvatar MACA, MetaX MUSA, etc.)
                from sglang.srt.platforms import current_platform

                CudaBackend._available = (
                    getattr(current_platform, "_vendor_name", None) == "nvidia"
                )
            except (ImportError, Exception):
                CudaBackend._available = False
        return CudaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_cuda

        return silu_and_mul_cuda(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_cuda

        return rms_norm_cuda(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_cuda

        return rotary_embedding_cuda(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def rotary_embedding_with_kv_cache(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        fused_set_kv_buffer_arg,
        rotary_interleaved: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.rotary import rotary_embedding_with_kv_cache_cuda

        return rotary_embedding_with_kv_cache_cuda(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            fused_set_kv_buffer_arg,
            rotary_interleaved=rotary_interleaved,
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
        from .impl.topk import topk_cuda

        return topk_cuda(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):
        from .impl.fused_moe import fused_moe_cuda

        return fused_moe_cuda(obj, layer, dispatch_output)

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.gemma_rms_norm import gemma_rms_norm_cuda

        return gemma_rms_norm_cuda(obj, x, residual)

    def mrotary_embedding(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.mrotary_embedding import mrotary_embedding_cuda

        return mrotary_embedding_cuda(obj, positions, query, key)

    def mrotary_embedding_with_kv_cache(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.mrotary_embedding import mrotary_embedding_with_kv_cache_cuda

        return mrotary_embedding_with_kv_cache_cuda(
            obj, positions, query, key, fused_set_kv_buffer_arg
        )

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
        from .impl.fla import chunk_gated_delta_rule_cuda

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
        from .impl.fla import fused_recurrent_gated_delta_rule_cuda

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
        from .impl.fla import fused_recurrent_gated_delta_rule_packed_decode_cuda

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
