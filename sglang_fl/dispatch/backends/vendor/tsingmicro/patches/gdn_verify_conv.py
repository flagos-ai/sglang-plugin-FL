# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN conv update of MTP target-verify on txda: one draft token per kernel call.

EAGLE/MTP target verify calls ``causal_conv1d_update`` from
``sglang.srt.layers.attention.linear.gdn_backend`` with all draft tokens at once
(``x``: ``(batch, dim, draft_tokens)``) and ``intermediate_conv_window`` set, so
the conv window after every draft token is saved for the accept/commit step.
FlagTree's tsingmicro compiler cannot lower that Triton specialization
(``seqlen > 1`` with ``SAVE_INTERMEDIATE``): ``tsingmicro-opt`` aborts with
``MemRefType::getNumElements ... hasStaticShape()``. The plain single-token
decode variant compiles and is correct.

For chain verify (EAGLE ``topk == 1``: no ``retrieve_*`` tree tensors, no
``num_accept_tokens``, no ``cache_seqlens``) run the single-token decode kernel
on a private copy of each request's conv history, one draft token at a time,
and snapshot the window after every token into ``intermediate_conv_window``
(what upstream's ``SAVE_INTERMEDIATE`` stores). The persistent ``conv_state`` is
not touched: after verify SGLang commits the accepted step's window from
``intermediate_conv_window`` (``scatter_mamba_states_after_mtp_verify``).
Padding rows (``conv_state_indices == -1``) write to the discard row
(the last row of ``intermediate_conv_window``) without index-dependent control
flow. Tree verify and all other calls go to SGLang's function unchanged.

Needs dense (non-overlapping) conv windows; see ``eagle_runtime`` (dedup off).
Remove once FlagTree compiles the multi-token variant.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("gdn_verify_conv")

_MODULE = "sglang.srt.layers.attention.linear.gdn_backend"
_patched = False


def _on_txda(x) -> bool:
    return x.device.type == "txda"


def _make_per_token(orig):
    import functools

    import torch

    @functools.wraps(orig)
    def causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        *,
        conv_state_indices=None,
        intermediate_conv_window=None,
        intermediate_state_indices=None,
        **kwargs,
    ):
        if (
            not _on_txda(x)
            or x.dim() != 3
            or intermediate_conv_window is None
            or conv_state_indices is None
            or intermediate_state_indices is None
            or any(value is not None for value in kwargs.values())
        ):
            return orig(
                x, conv_state, weight, bias, activation,
                conv_state_indices=conv_state_indices,
                intermediate_conv_window=intermediate_conv_window,
                intermediate_state_indices=intermediate_state_indices,
                **kwargs,
            )
        if x.shape[0] == 0:
            return torch.empty_like(x)

        width_minus_1 = weight.shape[-1] - 1
        source = conv_state_indices.long().clamp_min(0)
        destination = torch.where(
            conv_state_indices == -1,
            intermediate_conv_window.shape[0] - 1,
            intermediate_state_indices,
        ).long()
        history = conv_state.index_select(0, source).contiguous()  # (B, dim, state_len)
        outputs, snapshots = [], []
        for token in x.unbind(dim=-1):  # (B, dim)
            outputs.append(orig(token.contiguous(), history, weight, bias, activation))
            snapshots.append(history[..., :width_minus_1].clone())
        steps = x.shape[-1]
        intermediate_conv_window[:, :steps].index_copy_(
            0, destination, torch.stack(snapshots, dim=1)
        )
        return torch.stack(outputs, dim=1).transpose(1, 2)  # (B, dim, steps)

    causal_conv1d_update._txda_per_token_verify = True
    return causal_conv1d_update


def _apply() -> None:
    try:
        import sglang.srt.layers.attention.linear.gdn_backend as gdn

        orig = gdn.causal_conv1d_update
        if getattr(orig, "_txda_per_token_verify", False):
            return
        gdn.causal_conv1d_update = _make_per_token(orig)
        _log.applied("GDN verify causal_conv1d_update -> single-token updates on a private history (txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch gdn_backend.causal_conv1d_update: %s", exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_MODULE, _apply)
    _patched = True
