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
rms_norm_gated -> _layer_norm_fwd``, i.e. the native single-pass Triton
kernel.  This patch keeps it the only implementation GDN can reach:

1. rebinds ``qwen3_5.RMSNormGated`` so ``__init__`` constructs exactly
   ``layernorm_gated.RMSNorm`` (regardless of what another plugin or back end
   may have put there), and
2. wraps ``layernorm_gated.RMSNorm.forward`` so the call is observably routed
   through the upstream class -- one INFO line on the first call, one DEBUG
   line per call (for "is it really called per layer per step" checks).

By default the wrapper only logs and then calls the original ``forward``
(numerics untouched); disable the whole patch with ``SGLANG_FL_GDN_NORM=0``.

``SGLANG_FL_GDN_NORM_REF=1`` additionally replaces the numerics with the
module's own pure-PyTorch reference (``rms_norm_ref``).  The native Triton
kernel is the documented source of the TP=8 ``'!'``/NaN output on TXDA
(docs/sglang-plugin-fl-tp8-nan-fix.md), so this is the switch to use when
hunting that failure.
"""

import os

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("gdn_norm")

_ENABLED_KEY = "SGLANG_FL_GDN_NORM"
_REF_KEY = "SGLANG_FL_GDN_NORM_REF"
_TRUTHY = ("1", "true", "yes", "on")

_patched = False
_forward_orig = None
_calls = 0
_layernorm_gated = None


def _enabled() -> bool:
    return os.getenv(_ENABLED_KEY, "0").lower() in _TRUTHY


def _ref_enabled() -> bool:
    return os.getenv(_REF_KEY, "0").strip().lower() in _TRUTHY


def _ref_applicable(norm) -> bool:
    """The reference covers exactly the config GatedDeltaNet constructs."""
    return (
        norm.group_size is None
        and norm.norm_before_gate
        and norm.activation == "swish"
        and norm.bias is None
    )


def _forward(self, x, z=None):
    """Replacement for ``layernorm_gated.RMSNorm.forward``."""
    global _calls
    _calls += 1
    if _calls == 1:
        _log.info(
            "GatedDeltaNet gated norm -> layernorm_gated.RMSNorm "
            "(x=%s z=%s dtype=%s, ref=%s)",
            tuple(x.shape),
            None if z is None else tuple(z.shape),
            x.dtype,
            _ref_enabled(),
        )
    else:
        _log.debug(
            "gated norm call #%d x=%s z=%s",
            _calls,
            tuple(x.shape),
            None if z is None else tuple(z.shape),
        )
    if _ref_enabled() and _ref_applicable(self):
        return _layernorm_gated.rms_norm_ref(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )
    return _forward_orig(self, x, z)


def patch() -> None:
    """Pin GDN's gated norm to ``layernorm_gated.RMSNorm``.  Idempotent."""
    global _patched, _forward_orig, _layernorm_gated
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
    _layernorm_gated = lg

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
    _log.applied(
        "GatedDeltaNet gated norm pinned to layernorm_gated.RMSNorm "
        "(numerics=%s)",
        "pure-PyTorch reference" if _ref_enabled() else "native Triton kernel",
    )
