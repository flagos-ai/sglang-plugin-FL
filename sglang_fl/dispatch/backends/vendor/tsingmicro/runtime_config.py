# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang server-argument defaults for TsingMicro TXDA (validated on SGLang 0.5.18).

Called from ``PlatformFL.apply_server_args_defaults()``, which SGLang runs
from ``ServerArgs.__post_init__`` *before* it resolves its own memory and
chunked-prefill defaults. A field is therefore only filled when it still holds
SGLang's declared default, so explicit user choices always win.

These are the values the Qwen3.6 27B / 35B-A3B offline, concurrent, serving
and TP8/PP2 cross-node runs were validated with. Model-dependent values
(``context_length``, ``chunked_prefill_size``) stay in the launch scripts and
test cases.
"""

import logging

logger = logging.getLogger(__name__)

# field -> (SGLang 0.5.18 declared default, TXDA default)
_TXDA_DEFAULTS = {
    # TXDA kernels are validated in bfloat16 only.
    "dtype": ("auto", "bfloat16"),
    # Radix (prefix) caching is not validated on TXDA; the hybrid GDN models
    # use ChunkCache with the txda KV-release hardening (patches/kv_leak_fix).
    "disable_radix_cache": (False, True),
    # First-use Triton compilation on TXDA takes minutes per kernel family;
    # SGLang's 300 s watchdog fires during a cold-cache warmup.
    "watchdog_timeout": (300, 3600),
    # Vision-tower attention: the Triton kernel; the default selection would
    # pick a CUDA-only backend.
    "mm_attention_backend": (None, "triton_attn"),
    # The fast (torchvision) image processor is not validated on TXDA.
    # (= the deprecated ``--disable-fast-image-processor``.)
    "image_processor_backend": ("auto", "pil"),
}


def apply_tsingmicro_runtime_defaults(server_args) -> None:
    if getattr(server_args, "device", None) in (None, "cuda"):
        server_args.device = "txda"

    applied = []
    for field, (sglang_default, txda_default) in _TXDA_DEFAULTS.items():
        if not hasattr(server_args, field):
            continue  # field absent in this SGLang version
        if getattr(server_args, field) == sglang_default:
            setattr(server_args, field, txda_default)
            applied.append(f"{field}={txda_default}")
    if applied:
        logger.info("TXDA server-arg defaults: %s", ", ".join(applied))
