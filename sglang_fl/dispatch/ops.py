# Backend base interface for operator implementations.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


class FLBackendBase(ABC):
    """
    Abstract base class for FL operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.

    All operator methods receive `obj` as the first argument — this is
    the framework's nn.Module instance (e.g., RMSNorm, SiluAndMul).
    The `obj` provides access to weights, config, etc.

    Interface contract for `obj`:
      - silu_and_mul: no attributes needed from obj
      - rms_norm: obj.weight, obj.variance_epsilon
      - rotary_embedding: no attributes needed (all passed as explicit args)
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available in the current environment."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name string (e.g., "flagos", "ascend", "reference")."""
        pass

    @property
    def vendor(self) -> Optional[str]:
        """Vendor name (required for VENDOR kind backends)."""
        return None

    # ==================== Activation Operators ====================

    @abstractmethod
    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling nn.Module (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        pass

    # ==================== Normalization Operators ====================

    @abstractmethod
    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling nn.Module (provides obj.weight, obj.variance_epsilon)
            x: Input tensor
            residual: Optional residual tensor to fuse

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual provided
        """
        pass

    # ==================== Position Embedding Operators ====================

    @abstractmethod
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
        """
        Apply rotary position embedding.

        Args:
            obj: The calling nn.Module (for interface consistency)
            query: Query tensor [num_tokens, num_heads, head_dim]
            key: Key tensor [num_tokens, num_kv_heads, head_dim]
            cos: Cosine cache [max_seq_len, rotary_dim]
            sin: Sine cache [max_seq_len, rotary_dim]
            position_ids: Position indices [num_tokens]
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        pass

    @abstractmethod
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
        """
        Fused rotary position embedding + KV cache write.

        Single kernel applies RoPE to q/k, then writes rotated k and v to KV cache.
        This avoids a separate memory round-trip for KV cache store.

        Args:
            obj: The calling nn.Module (for interface consistency)
            query: Query tensor [num_tokens, num_heads, head_dim]
            key: Key tensor [num_tokens, num_kv_heads, head_dim]
            cos: Cosine cache [max_seq_len, rotary_dim]
            sin: Sine cache [max_seq_len, rotary_dim]
            position_ids: Position indices [num_tokens]
            fused_set_kv_buffer_arg: FusedSetKVBufferArg or dict containing:
                - value: V tensor [num_tokens, num_v_heads * head_dim]
                - k_buffer/key_cache: KV cache key buffer
                - v_buffer/value_cache: KV cache value buffer
                - cache_loc/slot_mapping: indices for cache write
            rotary_interleaved: Whether to use interleaved rotary

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        pass
