# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the per-pair conv-window commit kernel after MTP verify on txda.

After EAGLE/MTP target verify, ``scatter_mamba_states_after_mtp_verify``
(``sglang.kernels.ops.mamba.mamba_state_scatter_triton``) copies the conv window
of each request's last accepted draft step into the persistent conv state.
When ``_conv_multi_eligible`` holds (bf16, contiguous, <= 4 conv types: the
Qwen3.6 case) it uses the single-launch ``fused_conv_window_scatter_multi``
kernel. On txda that kernel writes wrong conv states (single-card check against
a torch reference: max abs diff ~6 on the committed slots), so every accepted
step corrupts the GDN conv state and the generation degrades into garbage after
the first verify. The per-pair ``fused_conv_window_scatter_with_mask`` kernel,
SGLang's own fallback in the same function, matches the reference exactly.

Make ``_conv_multi_eligible`` return False on txda (the function looks the name
up in its module globals, so rebinding the module attribute is enough). The ssm
state commit is unaffected (matches the reference).

Remove once the multi kernel is correct on the TXDA Triton backend.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("mamba_commit_scatter")

_MODULE = "sglang.kernels.ops.mamba.mamba_state_scatter_triton"
_patched = False


def _conv_multi_not_eligible(pairs) -> bool:
    return False


def _apply() -> None:
    try:
        import sglang.kernels.ops.mamba.mamba_state_scatter_triton as m

        if not hasattr(m, "_conv_multi_eligible"):
            _log.skipped("no _conv_multi_eligible in this SGLang")
            return
        m._conv_multi_eligible = _conv_multi_not_eligible
        _log.applied("MTP conv-window commit -> per-pair scatter kernel (multi kernel wrong on txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch mamba_state_scatter_triton: %s", exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_MODULE, _apply)
    _patched = True
