# TXDA FLA (Flash Linear Attention) operator implementations.
#
# TODO: Replace SGLang's original triton kernels with torch_txda native kernel
# once verified on hardware. Current behavior: SGLang's original triton kernels.

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)

logger = logging.getLogger(__name__)


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn


def _sync_device(device: torch.device) -> None:
    """Device-agnostic barrier (``torch.txda.synchronize`` on TXDA)."""
    mod = getattr(torch, device.type, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def _check_cu_seqlens(values, total_tokens: Optional[int]) -> Optional[str]:
    """Return a human-readable reason why ``values`` is not valid varlen metadata.

    ``values`` is the host copy of ``cu_seqlens``: the cumulative token offsets
    of the packed prefill batch, ``[0, len(s0), len(s0)+len(s1), ...]``.
    """
    if len(values) < 2:
        return f"expected at least 2 entries (one sequence), got {len(values)}"
    if values[0] != 0:
        return f"first entry must be 0, got {values[0]}"
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            return (
                f"not monotonically non-decreasing at index {i}: "
                f"{values[i - 1]} -> {values[i]}"
            )
    if total_tokens is not None and values[-1] != total_tokens:
        return (
            f"last entry must equal the packed token count {total_tokens}, "
            f"got {values[-1]}"
        )
    return None


def _validated_cu_seqlens(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor,
    head_first: bool,
) -> torch.Tensor:
    """Verify the varlen metadata and return it as a fresh device tensor.

    SGLang builds this tensor on device (``query_start_loc``) and every FLA
    kernel plus the chunk-grid sizing in flag_gems is derived from it.  When one
    entry is stale/garbage the kernels silently compute on the wrong sequence
    boundaries -- or, for a negative length, downstream sizing aborts with an
    unrelated-looking error ("Trying to create tensor with negative dimension
    -32" from ``torch.arange`` in flag_gems' ``prepare_chunk_indices``).

    The tensor only has ``n_seq + 1`` entries, so validating it on the host is
    a single small device-to-host copy.  If it does not validate, re-read after
    a device barrier (recovers a copy that had not landed yet) and then, for the
    one repairable case -- a corrupted *total* entry, which is the element
    SGLang computes with a scalar add followed by a scalar store -- restore the
    total from the batch itself.

    Returns a freshly built tensor so the kernels read exactly the values that
    were validated here.
    """
    # Packed-token count for this batch, from the q layout:
    #   [T, H, K] (3D, head_first=False), [H, T, K] (3D, head_first=True),
    #   [B, T, H, K] (4D, head_first=False), [B, H, T, K] (4D, head_first=True).
    if q.ndim >= 4:
        total_tokens = int(q.shape[0]) * int(q.shape[2] if head_first else q.shape[1])
    elif head_first:
        total_tokens = int(q.shape[1])
    else:
        total_tokens = int(q.shape[0])

    def _read():
        return [int(v) for v in cu_seqlens.detach().to("cpu").reshape(-1).tolist()]

    values = _read()
    problem = _check_cu_seqlens(values, total_tokens)
    if problem is not None:
        # A device-to-host copy that raced with the writer, or a buffer that was
        # overwritten, both show up here.  Drain the stream and look again.
        _sync_device(cu_seqlens.device)
        retry = _read()
        if retry != values:
            logger.warning(
                "[TXDA_FLA] cu_seqlens re-read differed after device sync: %s -> %s",
                values,
                retry,
            )
            values, problem = retry, _check_cu_seqlens(retry, total_tokens)

    if problem is not None:
        # The cumulative-total entry is the last thing SGLang writes, and it is
        # the only entry computed through a device scalar (an int32 add read
        # back on the host).  If everything else is consistent, the batch itself
        # tells us what that entry must be, so repair instead of failing.
        body = values[:-1]
        body_ok = (
            total_tokens is not None
            and len(values) >= 2
            and values[0] == 0
            and all(body[i] <= body[i + 1] for i in range(len(body) - 1))
            and body[-1] <= total_tokens
        )
        if body_ok:
            logger.warning(
                "[TXDA_FLA] corrupted cu_seqlens total repaired: %s -> %s "
                "(packed_tokens=%d, head_first=%s)",
                values,
                body + [total_tokens],
                total_tokens,
                head_first,
            )
            values = body + [total_tokens]
            problem = None

    if problem is not None:
        raise RuntimeError(
            "invalid cu_seqlens for chunk_gated_delta_rule: "
            f"{problem} (values={values}, head_first={head_first}); "
            "this tensor is SGLang's per-batch varlen metadata and the FLA "
            "kernels cannot run on it"
        )

    return torch.tensor(values, dtype=torch.long, device=q.device)


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

    # Validate the varlen metadata and hand the kernel a freshly built copy of
    # it (see _validated_cu_seqlens).  It is only n_seq + 1 entries, so this
    # costs one small D2H per GDN layer -- nothing against a chunked prefill.
    if cu_seqlens is not None:
        cu_seqlens = _validated_cu_seqlens(q, cu_seqlens, head_first)

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
