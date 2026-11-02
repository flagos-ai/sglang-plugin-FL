# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin GatedDeltaNet's gated RMSNorm to ``fla.layernorm_gated.RMSNorm``.

Qwen3.5/3.6 GatedDeltaNet reaches the gated norm through the *module-level*
name ``sglang.srt.models.qwen3_5.RMSNormGated``::

    # qwen3_5.py:42
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    # qwen3_5.py:233 (GatedDeltaNet.__init__)
    self.norm = RMSNormGated(self.head_v_dim, eps=..., group_size=None,
                             norm_before_gate=True, activation="swish")

    # qwen3_5.py:518 (GatedDeltaNet.forward)
    core_attn_out = self.norm(core_attn_out, z)

That class is a thin wrapper over ``layernorm_fn -> LayerNormFn.apply ->
rms_norm_gated -> _layer_norm_fwd``, i.e. the native single-pass Triton kernel.
This patch keeps it the only implementation GDN can reach:

1. rebinds ``qwen3_5.RMSNormGated`` so ``__init__`` constructs exactly
   ``layernorm_gated.RMSNorm`` (regardless of what another plugin or back end
   may have put there), and
2. wraps ``layernorm_gated.RMSNorm.forward`` so the call is observably routed
   through the upstream class -- one INFO line on the first call, one DEBUG
   line per call (for "is it really called per layer per step" checks).

The wrapper only logs and then calls the original ``forward``; numerics are
untouched.  Off by default: set ``SGLANG_FL_GDN_NORM=1`` to enable.
"""

import os

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("gdn_norm")

_patched = False
# Original ``RMSNorm.forward``; the wrapper calls through to it unchanged.
_forward_orig = None
# One-shot INFO on the first call, DEBUG afterwards (per layer, per step).
_calls = 0

_ENABLED_KEY = "SGLANG_FL_GDN_NORM"


def _enabled() -> bool:
    return os.getenv(_ENABLED_KEY, "0").lower() in ("1", "true", "yes", "on")


def _forward(self, x, z=None):
    """Replacement for ``layernorm_gated.RMSNorm.forward``."""
    global _calls
    _calls += 1
    if _calls == 1:
        _log.info(
            "GatedDeltaNet gated norm -> layernorm_gated.RMSNorm (x=%s z=%s dtype=%s)",
            tuple(x.shape),
            None if z is None else tuple(z.shape),
            x.dtype,
        )
    else:
        _log.debug(
            "gated norm call #%d x=%s z=%s",
            _calls,
            tuple(x.shape),
            None if z is None else tuple(z.shape),
        )
    return _forward_orig(self, x, z)


def patch() -> None:
    """Pin GDN's gated norm to ``layernorm_gated.RMSNorm``. Idempotent."""
    global _patched, _forward_orig
    if _patched:
        return
    if not _enabled():
        _log.skipped("disabled (set %s=1 to enable)", _ENABLED_KEY)
        return

    try:
        from sglang.srt.layers.attention.fla import layernorm_gated as lg
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot import layernorm_gated: %s", exc)
        return

    cls = lg.RMSNorm
    _forward_orig = cls.forward
    cls.forward = _forward

    try:
        import sglang.srt.models.qwen3_5 as qwen3_5

        qwen3_5.RMSNormGated = cls
    except Exception as exc:  # noqa: BLE001
        # The class-level wrap above still routes the call; the rebind only
        # matters if a different symbol had been installed on the model module.
        _log.warning("cannot rebind qwen3_5.RMSNormGated: %s", exc)

    _patched = True
    _log.applied("GatedDeltaNet gated norm pinned to layernorm_gated.RMSNorm")
