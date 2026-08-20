# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch sglang's fused MoE runner for txda platform compatibility.

Patches applied:

  1. ``moe_align_block_size`` — replace the function reference in
     ``fused_moe`` with our txda‑safe copy that unconditionally imports
     ``sgl_kernel.moe_align_block_size`` (bypassing the
     ``is_cuda/hip/xpu/musa`` guard).
     Replaces ``moe_runner/triton_utils/moe_align_block_size.py:14-16``.

  2. ``fused_moe`` — force ``_has_vllm_ops = False`` (disable vllm import
     which is not available on txda).
     Replaces ``moe_runner/triton_utils/fused_moe.py:73-82``.

  3. ``fused_moe`` — replace ``inplace_fused_experts`` and
     ``outplace_fused_experts`` with plain wrappers that call
     ``fused_experts_impl`` directly, bypassing ``@register_custom_op``
     which may fail on txda devices.
     Replaces ``moe_runner/triton_utils/fused_moe.py:87,148``.

  4. ``UnquantizedFusedMoEMethod.apply`` — route the MoE apply entry
     through ``self.forward()`` (dispatch system) so the txda vendor
     backend can intercept fused_moe via the standard dispatch bridge,
     instead of calling ``forward_cuda`` directly.
     Replaces ``sglang/srt/layers/quantization/unquant.py``.
"""

import functools

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("fused_moe")

_patched = False


def _patch_moe_align_block_size() -> None:
    """Replace ``moe_align_block_size`` in the fused_moe module with our
    txda‑safe version.

    Sglang's upstream module guards ``from sgl_kernel import
    moe_align_block_size`` behind ``is_cuda/hip/xpu/musa``, all of which
    evaluate to ``False`` on txda because ``torch_txda`` does not set
    ``torch.version.cuda``.  This causes a ``NameError`` at runtime when
    ``_prepare_fused_moe_run`` calls the (unimported) symbol.

    Instead of trying to re‑create the original import conditions we
    simply **swap** the reference that ``fused_moe.py`` holds for
    ``moe_align_block_size`` with our own copy that unconditionally
    imports the kernel.
    """
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fm
    except Exception as exc:
        _log.failed("failed to import fused_moe module: %s", exc)
        return

    try:
        from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.fused_moe.moe_align_block_size import (
            moe_align_block_size as _txda_moe_align,
        )
    except Exception as exc:
        _log.failed("failed to import local moe_align_block_size: %s", exc)
        return

    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size as _mas

        _mas.moe_align_block_size = _txda_moe_align
    except Exception:
        pass

    _fm.moe_align_block_size = _txda_moe_align
    _log.applied("moe_align_block_size: replaced with txda-safe version")


def _patch_fused_moe_vllm() -> None:
    """Force ``_has_vllm_ops = False`` to prevent vllm_ops usage on txda.

    On txda, the original ``if not _is_cuda and not _is_hip and not _is_xpu``
    guard is True, so vllm is attempted on import.  Overriding the flag after
    import prevents the fallback path from calling vllm ops, which would fail
    on txda hardware.
    """
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fm

        _fm._has_vllm_ops = False
        _log.applied("fused_moe: _has_vllm_ops forced to False on txda")
    except Exception as exc:
        _log.failed("failed to patch fused_moe vllm_ops: %s", exc)


def _patch_unquant_apply() -> None:
    """Route ``UnquantizedFusedMoEMethod.apply`` through ``self.forward()``.

    The original ``apply()`` calls ``self.forward_cuda()`` directly,
    bypassing the dispatch bridge.  Replacing it with ``self.forward()``
    sends it through ``MultiPlatformOp.dispatch_forward()`` → AROUND hook →
    ``fused_moe_bridge`` → ``call_op("fused_moe", ...)``, allowing the
    txda vendor backend to intercept fused_moe like any other op.

    The original function is preserved as ``_orig_apply`` on the class
    for potential restoration.
    """
    try:
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
        )

        _orig_apply = UnquantizedFusedMoEMethod.apply

        @functools.wraps(_orig_apply)
        def _patched_apply(self, layer, dispatch_output):
            return self.forward(
                layer=layer,
                dispatch_output=dispatch_output,
            )

        UnquantizedFusedMoEMethod.apply = _patched_apply
        UnquantizedFusedMoEMethod._orig_apply = _orig_apply
        _log.applied(
            "UnquantizedFusedMoEMethod.apply patched → self.forward()"
        )
    except Exception as exc:
        _log.failed("failed to patch UnquantizedFusedMoEMethod.apply: %s", exc)


def patch() -> None:
    """Apply all fused MoE monkey-patches for txda platform.

    Idempotent: safe to call multiple times.  Non-txda platforms are no-ops.
    """
    global _patched
    if _patched:
        return

    if not _is_txda():
        _log.skipped("txda not available — fused_moe patches skipped")
        return

    _patch_moe_align_block_size()
    _patch_fused_moe_vllm()
    _patch_unquant_apply()

    _patched = True
