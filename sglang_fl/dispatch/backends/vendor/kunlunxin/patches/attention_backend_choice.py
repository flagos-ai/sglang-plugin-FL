"""Register 'kunlunxin' as a valid ``--attention-backend`` CLI choice.

sglang's ``sglang.launch_server`` CLI validates ``--attention-backend`` against
the static ``ATTENTION_BACKEND_CHOICES`` list at parse time. Our OOT backend
name is not in that list, so ``--attention-backend kunlunxin`` (used by the
multinode run scripts) fails argparse with "invalid choice". ``launch_server``
calls ``load_plugins()`` before ``prepare_server_args()``, so extending the list
here (during plugin load) makes the CLI accept it. The single-node Engine path
builds ServerArgs from kwargs and does not validate choices, so this is a
harmless no-op there.
"""

import logging

logger = logging.getLogger(__name__)
_patched = False


def patch_attention_backend_choice():
    global _patched
    if _patched:
        return
    _patched = True

    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    if "kunlunxin" not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices(["kunlunxin"])
