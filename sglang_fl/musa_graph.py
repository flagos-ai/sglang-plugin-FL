# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MUSA decode graph with a replay guard for QSA's all-blocks specialization."""

import logging

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

logger = logging.getLogger(__name__)


def qsa_context_supported(seq_lens_cpu, limit):
    if limit is None:
        return True
    if seq_lens_cpu is None or len(seq_lens_cpu) == 0:
        return False
    if hasattr(seq_lens_cpu, "device") and seq_lens_cpu.device.type != "cpu":
        return False
    lengths = seq_lens_cpu.tolist() if hasattr(seq_lens_cpu, "tolist") else seq_lens_cpu
    return max(lengths) <= limit


class MusaDecodeGraphRunner(DecodeCudaGraphRunner):
    def __init__(self, model_runner, *args, **kwargs):
        super().__init__(model_runner, *args, **kwargs)
        limits = [
            m._fl_graph_all_blocks_limit
            for m in model_runner.model.modules()
            if hasattr(m, "_fl_graph_all_blocks_limit")
        ]
        self._qsa_all_blocks_limit = min(limits) if limits else None
        logger.info(
            "MUSA graph QSA all-blocks replay limit: %s; longer contexts use eager",
            self._qsa_all_blocks_limit,
        )

    def capture(self):
        from sglang_fl.models.qwen4_communication import groups

        communicators = [
            getattr(group, "_fl_qwen38_custom_ar", None) for group in groups()
        ]
        if any(comm is not None for comm in communicators):
            self.backend.cleanup()
            for comm in communicators:
                if comm is not None:
                    comm.reset_graph_buffers()
        super().capture()

    def can_run_graph(self, forward_batch):
        if not qsa_context_supported(
            forward_batch.seq_lens_cpu, self._qsa_all_blocks_limit
        ):
            return False
        return super().can_run_graph(forward_batch)
