# Copyright (c) 2026 BAAI. All rights reserved.
"""Hygon packed decode FLA adapter using SGLang's public launcher."""

import torch

from .fla_compat import original_fla


def fused_recurrent_gated_delta_rule_packed_decode_hcu(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the public launcher's validation and in-place state contract.

    The SGLang 0.5.18 launcher retains the HCU wrapper's BV=32 cap, one warp
    and three stages. Delegation avoids importing the removed SRT namespace
    or maintaining a second copy of the private Triton launch ABI.
    """
    return original_fla("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
