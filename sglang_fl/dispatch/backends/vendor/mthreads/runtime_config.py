# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Runtime fallbacks validated with SGLang 0.5.18 on MUSA."""

import logging

logger = logging.getLogger(__name__)

_QWEN35_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


_BUGGY_GDN_BETA_EXPRESSION = (
    "beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
)


def needs_gdn_packed_decode_backport(sglang_version: str) -> bool:
    """Return whether this SGLang release needs the packed GDN kernel backport."""
    return sglang_version.split("+", maxsplit=1)[0] == "0.5.18"


def install_gdn_packed_decode_backport(
    kernel_module, fixed_kernel, sglang_version: str
) -> bool:
    """Replace the lossy 0.5.18 packed kernel while retaining its fast path.

    The packed kernel rounds ``sigmoid(beta)`` through BF16 before updating the
    persistent FP32 SSM state. That error accumulates one token at a time and
    makes normal decode diverge from MTP target verification, whose generic
    recurrent kernel keeps beta in FP32. Upstream tracks the same defect in
    sgl-project/sglang#38975 and fixes it in #38977.

    Check both the pinned release and the exact defective source so a backported
    or future SGLang build keeps its own implementation.
    """
    if not needs_gdn_packed_decode_backport(sglang_version):
        return False

    name = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
    installed_kernel = getattr(kernel_module, name, None)
    if _BUGGY_GDN_BETA_EXPRESSION not in getattr(installed_kernel, "src", ""):
        return False

    setattr(kernel_module, name, fixed_kernel)
    return True


def apply_musa_runtime_defaults(server_args) -> None:
    # Older SGLang does not expose this hybrid-cache configuration.
    if not hasattr(server_args, "mamba_radix_cache_strategy"):
        return
    speculative = getattr(server_args, "speculative_algorithm", None) not in (
        None,
        "NONE",
    )
    if not speculative:
        return

    architectures = server_args.get_model_config().hf_config.architectures
    arch = architectures[0] if architectures else None
    qwen_mtp = speculative and arch in _QWEN35_ARCHS
    if not qwen_mtp:
        return

    # MTP graph replay stalled waiting for result.copy_done.
    # Synchronous scheduling is the validated fallback;
    # keep graph capture/replay and the original request concurrency enabled.
    if not server_args.disable_overlap_schedule:
        logger.warning(
            "MUSA Qwen3.5/3.6 MTP uses synchronous scheduling to avoid the "
            "0.5.18 overlap watchdog failure; graph settings are retained",
        )
        server_args.disable_overlap_schedule = True
