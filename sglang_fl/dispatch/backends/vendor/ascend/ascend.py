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
        raise NotImplementedError(
            "rotary_embedding_with_kv_cache is not implemented in Ascend backend. "
            "Requires a fused RoPE + KV cache write kernel from torch_npu."
        )
