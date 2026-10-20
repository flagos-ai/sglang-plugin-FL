# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

logger = logging.getLogger(__name__)
_patched = False


def patch_clamp_position() -> None:
    global _patched
    if _patched:
        return

    from sglang.srt.model_executor import forward_batch_info

    forward_batch_info.clamp_position = forward_batch_info._clamp_position_native
    _patched = True
    logger.info("iluvatar clamp_position routed to torch native")
