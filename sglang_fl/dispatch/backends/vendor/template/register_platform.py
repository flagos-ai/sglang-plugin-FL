"""Template: OOT attention-backend registration for new hardware vendors.

This file is a working skeleton — copy the parent directory to
``sglang_fl/dispatch/backends/vendor/<your_vendor>/`` and adapt.

Wiring overview
---------------
``PlatformFL.init_backend()`` (in sglang_fl/platform.py) auto-imports
``sglang_fl.dispatch.backends.vendor.<vendor_name>.register_platform`` when
sglang activates an OOT platform. Importing this module executes the
``@register_attention_backend`` decorator below, which inserts your backend
creator into sglang's ``ATTENTION_BACKENDS`` dict. After that,
``ModelRunner._get_attention_backend_from_str(name)`` can resolve it.

To onboard your vendor, do all four:
  1. Make sure FlagGems' ``DeviceDetector`` returns your vendor name
     (e.g. "myvendor") on your hardware.
  2. Add ``"myvendor": "myvendor_oot"`` to ``_ATTN_BACKEND_MAP`` in
     ``sglang_fl/platform.py``.
  3. Rename this directory from ``template/`` to ``myvendor/``.
  4. Implement ``TemplateOOTAttnBackend`` in ``impl/attention_backend.py``
     (and rename it to ``MyvendorOOTAttnBackend``).

Note: the ``template`` vendor name itself will never match a real
DeviceDetector result, so ``init_backend()`` will silently skip this
module in production — it exists purely as documentation.
"""

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


@register_attention_backend("template_oot")
def _create_template_oot_backend(runner):
    """Creator for 'template_oot' in sglang's ATTENTION_BACKENDS registry.

    Called by ModelRunner._get_attention_backend_from_str when
    ``server_args.attention_backend == "template_oot"``. Construct your
    AttentionBackend subclass here and return the instance.

    Defer heavy imports (vendor SDK, CUDA-only headers, etc.) to the body
    of this function so the module stays importable on any host.
    """
    from sglang_fl.dispatch.backends.vendor.template.impl.attention_backend import (
        TemplateOOTAttnBackend,
    )

    return TemplateOOTAttnBackend(runner)
