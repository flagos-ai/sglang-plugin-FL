# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable queue assignment for a group of PD destination sessions."""

import threading


class PeerQueueShards:
    """Keep each destination group on one FIFO and spread new groups evenly."""

    def __init__(self, queue_count):
        if queue_count < 1:
            raise ValueError("At least one transfer queue is required")
        self._queue_count = queue_count
        self._peers = {}
        self._lock = threading.Lock()

    def select(self, sessions):
        key = tuple(sorted(sessions))
        if not key:
            return 0
        with self._lock:
            if key not in self._peers:
                self._peers[key] = len(self._peers) % self._queue_count
            return self._peers[key]
