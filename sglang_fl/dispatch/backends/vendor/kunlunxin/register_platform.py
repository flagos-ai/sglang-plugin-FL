"""OOT attention-backend registration for the kunlunxin vendor.

Auto-imported by ``PlatformFL.init_backend()`` when FlagGems reports
vendor_name == "kunlunxin". Importing this module registers the
``kunlunxin`` attention backend into sglang's ATTENTION_BACKENDS dict.
"""

from sglang.srt.layers.attention.attention_registry import register_attention_backend


@register_attention_backend("kunlunxin")
def _create_kunlunxin_backend(runner):
    from sglang_fl.dispatch.backends.vendor.kunlunxin.impl.attention_backend import (
        KunlunxinBackend,
    )

    return KunlunxinBackend(runner)
