# TXDA FLA (Flash Linear Attention) operator implementations.
#
# TODO: Replace SGLang's original triton kernels with torch_txda native kernel
# once verified on hardware. Current behavior: SGLang's original triton kernels.

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn

def chunk_gated_delta_rule_txda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """chunk_gated_delta_rule on TXDA via flaggems ``_tsingmicro`` backend.

    Adapts between SGLang's and flaggems' recurrent-state conventions:

    * SGLang stores the state as ``[..., V, K]`` (``ssm_states`` is
      ``[num_slots, H, V, K]``) and writes the final state back in-place into
      ``initial_state[initial_state_indices]`` (``INPLACE_UPDATE``).
    * flaggems stores the state as ``[..., K, V]`` and returns the final state as
      a fresh tensor.

    So here we pre-index + transpose on the way in, transpose back on the way
    out, and write the final state back into the caller's ``initial_state``.

    When the installed flag_gems predates this op, fall back to SGLang's own
    ``chunk_gated_delta_rule`` (which does the same adaptation itself).
    """
    txda_chunk_gated_delta_rule = load_txda_op("chunk_gated_delta_rule")
    if txda_chunk_gated_delta_rule is None:
        return _original("chunk_gated_delta_rule")(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens, head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    # flaggems requires g/beta/initial_state to share q's dtype, but SGLang's
    # GDN gating (fused_gdn_gating) and the state pool may be a different dtype
    # (e.g. fp32).  Align them to q's compute dtype before the call.
    #
    # q/k/v are left NON-contiguous: SGLang produces them via torch.split +
    # view (strided), and a .contiguous() copy is very expensive on TXDA for
    # these large tensors.  The TXDA FLA kernels accept strided inputs (they
    # read the actual tensor strides), so no copy is needed.  g/beta/h0 are
    # small, so materializing them contiguous here is cheap.
    g = g.to(q.dtype).contiguous()
    beta = beta.to(q.dtype).contiguous()

    h0 = initial_state
    if initial_state is not None and initial_state_indices is not None:
        h0 = initial_state[initial_state_indices]  # [N, H, V, K]
    if h0 is not None:
        h0 = h0.transpose(-1, -2).to(q.dtype).contiguous()  # [N, H, K, V]

    if cu_seqlens is not None and cu_seqlens.dtype != torch.long:
        cu_seqlens = cu_seqlens.long()

    o, final_state = txda_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        BT=64,
        initial_state=h0,
        output_final_state=(initial_state is not None),
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Replicate SGLang's INPLACE_UPDATE: write the final state back into the
    # state pool, transposing [K, V] -> [V, K] to match ssm_states layout.
    if (
        initial_state is not None
        and initial_state_indices is not None
        and final_state is not None
    ):
        initial_state[initial_state_indices] = final_state.transpose(-1, -2).to(
            initial_state.dtype
        )

    # SGLang's return contract is (o, None, h), where the third element is the
    # per-chunk state [B, NT, H, V, K] used only for prefix-cache SSM tracking.
    # flaggems exposes the final state, not the per-chunk states, so tracking is
    # not wired here; return None for that slot.
    return o, None, None


def fused_recurrent_gated_delta_rule_txda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """fused_recurrent_gated_delta_rule — not yet implemented on TXDA. Current behavior: SGLang's original triton kernels."""
    return _original("fused_recurrent_gated_delta_rule")(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_txda(
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
    """fused_recurrent_gated_delta_rule_packed_decode — not yet implemented on TXDA. Current behavior: SGLang's original triton kernels."""
    return _original("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log,
        dt_bias=dt_bias, scale=scale,
        initial_state=initial_state, out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
