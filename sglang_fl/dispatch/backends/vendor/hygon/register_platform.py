"""Register the Hygon attention implementation."""

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


@register_attention_backend("hcu")
def _create_hcu_attention_backend(runner):
    """Create the HCU backend without importing Triton during registration."""
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the HCU attention backend."
    )

    from sglang_fl.dispatch.backends.vendor.hcu.impl.attention_backend import (
        HCUAttnBackend,
    )

    logger.info("Using HCU attention backend")
    return HCUAttnBackend(runner)
