"""GCU all-greedy decode fix: torch.argmax -> topk on gcu (torch_gcu 1.9.10).

torch_gcu 1.9.10's native argmax reduction kernel returns an OUT-OF-VOCAB
index on real decode logits (e.g. 153604 when vocab_size is 151936): the
reduction's unmasked tail block writes garbage that the kernel never clamps.
sglang's all-greedy fast path (``sglang/srt/layers/sampler.py``,
``Sampler.forward`` -> ``is_all_greedy``) picks the next token with
``torch.argmax(logits, -1)``, so that garbage index is emitted as a token id
-> an unmapped / stop token -> premature EOS after 2-3 decode steps.
``torch.topk(logits, 1)`` on the same tensor returns the correct index (the
sampling paths -- topk/multinomial based -- are correct on gcu), so greedy
selection is rerouted through topk.

Not compiler- or flag_gems-specific (both F and T decode paths show it): the
native kernel is simply wrong for argmax. The fix therefore replaces
``torch.argmax`` itself for gcu tensors, at the torch-module level -- the
same level the sampler reaches it. The module-level patch applies on
patch.py import, so every process that loads the vendor layer gets it
(matching the other enflame patches). Non-gcu tensors fall through to the
original ``torch.argmax`` unchanged, so other backends and CPU/meta tensors
are untouched. The wrapper reproduces ``torch.argmax``'s exact API (dim /
keepdim semantics, int64 indices), so call sites that were already correct
keep identical results.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

_applied = False

# torch_gcu registers a PrivateUse1 backend and aliases torch.cuda.* to gcu.
# Tensors can therefore report "gcu", the generic pre-rename type, or the
# aliased "cuda" type -- all of them are physically gcu in this vendor layer
# (this module is only ever imported when DeviceDetector says enflame).
_GCU_DEVICE_TYPES = ("gcu", "privateuseone", "cuda")

_orig_argmax: Optional[Callable] = None


def _argmax_via_topk(
    input: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> torch.Tensor:
    """``torch.argmax``-equivalent computed with ``torch.topk`` (gcu-safe).

    ``torch.topk(x, 1, dim=d)`` is the max over dim ``d`` with int64 indices,
    so the result matches ``torch.argmax`` shape-for-shape in all three
    signatures: ``dim=None`` flattens to a 0-dim index, ``keepdim=True`` keeps
    the reduced dim at size 1, ``keepdim=False`` drops it.
    """
    if dim is None:
        return torch.topk(input.reshape(-1), 1).indices.reshape(())
    indices = torch.topk(input, 1, dim=dim).indices
    if not keepdim:
        indices = indices.squeeze(dim)
    return indices


def _argmax_gcu_safe(
    input: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> torch.Tensor:
    """``torch.argmax`` that reroutes gcu tensors through the topk kernel."""
    try:
        if input.device.type in _GCU_DEVICE_TYPES:
            return _argmax_via_topk(input, dim=dim, keepdim=keepdim)
    except Exception as e:  # noqa: BLE001 - never let the compat path take sglang down
        logger.warning("gcu argmax->topk fallback failed (%r); using torch.argmax", e)
    return _orig_argmax(input, dim=dim, keepdim=keepdim)


def patch_sampler_greedy() -> None:
    """Replace ``torch.argmax`` with a gcu topk-backed wrapper (idempotent)."""
    global _applied, _orig_argmax
    if _applied:
        return
    try:
        _orig_argmax = torch.argmax
        torch.argmax = _argmax_gcu_safe
        _applied = True
        logger.info("gcu sampler-greedy patch applied (torch.argmax -> topk on gcu)")
    except Exception as e:  # noqa: BLE001
        logger.warning("gcu sampler-greedy patch failed: %r", e)


patch_sampler_greedy()
