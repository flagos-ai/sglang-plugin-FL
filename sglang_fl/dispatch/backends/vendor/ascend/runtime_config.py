# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend runtime defaults required for reproducible inference."""

import logging
import os


logger = logging.getLogger(__name__)


def apply_ascend_runtime_defaults(server_args) -> None:
    """Select deterministic HCCL reductions before worker communicators exist.

    CANN 8.5's A3 vector-core reductions can accumulate BF16 values in different
    orders across invocations. Those rounding differences change greedy decode
    even with fixed seeds and identical projection inputs. HCCL's deterministic
    option retains the native communication path, tensor dtype, graph execution
    and overlap scheduling, but may reduce throughput. Explicit environment
    choices, including an opt-out with ``false``, always take precedence.

    This callback is invoked only for Ascend and runs while server arguments are
    prepared, before spawning workers or initializing HCCL/FlagCX communicators.
    """
    if "HCCL_DETERMINISTIC" not in os.environ:
        os.environ["HCCL_DETERMINISTIC"] = "true"
        logger.info(
            "Ascend enables deterministic HCCL reductions by default; an explicit "
            "HCCL_DETERMINISTIC setting takes precedence."
        )
