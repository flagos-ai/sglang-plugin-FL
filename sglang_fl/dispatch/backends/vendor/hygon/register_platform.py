"""Register the Hygon attention implementation."""

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


def _create_hcu_attention_backend(runner):
    """Create the HCU backend without importing Triton during registration."""
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the HCU attention backend."
    )

    from sglang_fl.dispatch.backends.vendor.hygon.impl.attention_backend import (
        HCUAttnBackend,
    )

    logger.info("Using HCU attention backend")
    return HCUAttnBackend(runner)


def register_hcu_attention_backend() -> None:
    """Register the lazy factory and its CLI choice before argument parsing."""
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    # All three attention CLI arguments share this list. Use the public API to
    # mutate it in place, and avoid duplicate choices on repeated plugin setup.
    if "hcu" not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices(["hcu"])
    register_attention_backend("hcu")(_create_hcu_attention_backend)


# Keep direct PlatformFL.init_backend imports working as well as early plugin
# setup. The factory imports the HCU kernels only when a ModelRunner needs it.
register_hcu_attention_backend()
