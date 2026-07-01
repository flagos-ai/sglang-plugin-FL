# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA: OOT attention-backend registration for TsingMicro hardware.

Auto-imported by ``PlatformFL.init_backend()`` when sglang activates the
tsingmicro platform.  Importing this module executes the
``@register_attention_backend`` decorator below, which inserts the txda
backend creator into sglang's ``ATTENTION_BACKENDS`` dict.  After that,
``ModelRunner._get_attention_backend_from_str("txda_oot")`` can resolve it.

To onboard a custom TXDA attention kernel:
  1. Implement the backend in ``impl/attention_backend.py``.
  2. Update the creator below to return your backend instance.
  3. Set ``server_args.attention_backend = "txda_oot"`` in your launch script.
"""

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


@register_attention_backend("txda_oot")
def _create_txda_oot_backend(runner):
    """Creator for 'txda_oot' in sglang's ATTENTION_BACKENDS registry.

    Called by ``ModelRunner._get_attention_backend_from_str`` when
    ``server_args.attention_backend == "txda_oot"``.

    Defer heavy imports (vendor SDK, CUDA-only headers, etc.) to the body
    of this function so the module stays importable on any host.
    """
    # Default: use torch_native (PyTorch SDPA) for TXDA hardware.
    # Replace with a custom TXDA attention backend when available.
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)
