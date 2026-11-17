# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA: OOT attention-backend registration for TsingMicro hardware.

Auto-imported by ``PlatformFL.init_backend()`` when sglang activates the
tsingmicro platform.  Importing this module executes the
``@register_attention_backend`` decorator below, which inserts the txda
backend creator into sglang's ``ATTENTION_BACKENDS`` dict, where SGLang 0.5.18's
``attention_backend_setup._build_full_attention_backend_from_str`` resolves
``attention_backend="txda_oot"`` (the TXDA default, see ``sglang_fl/platform.py``).

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

    Called by SGLang's attention-backend setup when
    ``server_args.attention_backend == "txda_oot"``.

    Defer heavy imports (vendor SDK, CUDA-only headers, etc.) to the body
    of this function so the module stays importable on any host.
    """
    # Routes extend/prefill and decode to the flaggems _tsingmicro kernels;
    # metadata handling is inherited from TritonAttnBackend.
    from sglang_fl.dispatch.backends.vendor.tsingmicro.impl.attention_backend import (
        TxdaAttnBackend,
    )

    return TxdaAttnBackend(runner)


# Workers that SGLang starts without load_plugins() (sglang.bench_one_batch)
# load the full plugin before their first ModelConfig; see plugin_bootstrap.py.
# init_backend runs while ModelRunner is still being imported, so only the
# wrapper is installed here; loading the plugin waits for the first real
# model-config call.
from sglang_fl.dispatch.backends.vendor.tsingmicro.plugin_bootstrap import (  # noqa: E402
    install_model_config_bootstrap,
)

install_model_config_bootstrap()
