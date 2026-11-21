# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch sglang's Triton MoE runner for txda (SGLang 0.5.18).

Patch applied:

  ``moe_align_block_size`` — SGLang imports ``sgl_moe_align_block_size`` only
  on cuda/hip/xpu/musa (``moe_runner/triton_utils/moe_align_block_size.py``),
  so on txda the name is undefined and the Triton runner fallback raises
  ``NameError``. Rebind the name ``fused_moe.py`` calls to the FlagGems
  ``_tsingmicro`` implementation (``patches/fused_moe/moe_align_block_size.py``).
  The normal txda path does not reach this: the ``fused_moe`` dispatch op runs
  the vendor implementation (``impl/fused_moe.py``). Expert-parallel ids (-1)
  are not exercised on txda; upstream's 4th parameter is
  ``ignore_invalid_expert``, ours is ``expert_map``.

Removed for 0.5.18: forcing ``_has_vllm_ops = False`` (vllm is not installed,
so upstream's own import guard already yields False) and rerouting
``UnquantizedFusedMoEMethod.apply`` through ``self.forward()`` (upstream's
``apply`` does exactly that).
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("fused_moe")

_patched = False


def _patch_moe_align_block_size() -> None:
    """Rebind ``moe_align_block_size`` in fused_moe (and its defining module)."""
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

    _patched = True
