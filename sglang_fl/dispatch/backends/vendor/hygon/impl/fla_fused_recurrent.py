# Copyright (c) 2026 BAAI. All rights reserved.
"""Hygon recurrent FLA adapter for old and current SGLang public APIs."""

import inspect
from functools import lru_cache
from typing import Optional, Tuple

import torch

from .fla_compat import original_fla


@lru_cache(maxsize=None)
def _recurrent_parameters(fn):
    return inspect.signature(fn).parameters


def fused_recurrent_gated_delta_rule_hcu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Call the saved public function and preserve legacy state scheduling.

    SGLang 0.5.18 removed the scheduling arguments from this public function.
    Reject real scheduling requests when unsupported rather than silently
    updating the wrong state slots or dropping accepted-token information.
    """
    native = original_fla("fused_recurrent_gated_delta_rule")
    parameters = _recurrent_parameters(native)
    scheduling = {}
    for name, value in (
        ("ssm_state_indices", ssm_state_indices),
        ("num_accepted_tokens", num_accepted_tokens),
    ):
        if name in parameters:
            scheduling[name] = value
        elif value is not None:
            raise NotImplementedError(
                f"The installed SGLang recurrent FLA kernel does not support {name}"
            )
    return native(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **scheduling,
    )
