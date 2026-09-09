# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Runtime fallbacks validated with SGLang 0.5.18 on MUSA."""

import logging

logger = logging.getLogger(__name__)

_QWEN35_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


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
