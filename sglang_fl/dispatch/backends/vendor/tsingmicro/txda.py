# TXDA (TsingMicro) backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class TxdaBackend(Backend):
    """
    TXDA backend for operator implementations.

    Only ops whose TXDA implementation differs from the generic FlagOS /
    reference ones are registered (see register_ops.py); every other op
    resolves to default.flagos or reference.torch.
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

    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.gemma_rms_norm import gemma_rms_norm_txda
        return gemma_rms_norm_txda(obj, x, residual)

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
