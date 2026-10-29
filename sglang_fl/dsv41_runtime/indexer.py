# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Route DeepSeek V4.1 indexer calls through the configured backend policy."""


def producer(original, *args, **kwargs):
    from sglang_fl.dispatch import resolve_op

    return resolve_op("dsv41_indexer_producer")(original, *args, **kwargs)


def compact(original, *args, **kwargs):
    from sglang_fl.dispatch import resolve_op

    return resolve_op("dsv41_indexer_compact")(original, *args, **kwargs)
