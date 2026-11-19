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
        self._capture_metadata_graphs()

    def _capture_metadata_graphs(self):
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        backend = self.attn_backend
        if (
            not self.capture_forward_mode.is_decode()
            or not self.model_runner.spec_algorithm.is_none()
            or self.enable_two_batch_overlap
            or self.model_runner.lora_manager is not None
            or not isinstance(backend, HybridLinearAttnBackend)
            or not isinstance(backend.full_attn_backend, QwenSparseAttnBackend)
            or backend.linear_attn_backend.replayssm_write_pos_list is not None
        ):
            return
        self._metadata_glue = MetadataGlueGraph(self.device, strict=True)
        with self.model_context():
            for bs in self.capture_bs:
                batch, _, _ = self.capture_prepare(bs)
                key = self._glue_key(bs, str(self.capture_forward_mode), None)
                for _ in range(MetadataGlueGraph.NUM_WARMUP + 1):
                    self._init_replay_attn_metadata_from_view(
                        batch,
                        raw_bs=bs,
                        bs=bs,
                        num_tokens=bs,
                        glue_key=key,
                        attention_variant=None,
                    )
                if not self._metadata_glue.is_captured(key):
                    raise RuntimeError("Qwen3.8 metadata graph was not captured")
        self.device_module.synchronize()
        logger.info(
            "Qwen3.8 metadata graphs prepared before serving: %s", self.capture_bs
        )

    def _init_replay_attn_metadata(
        self,
        forward_batch,
        *,
        raw_bs,
        bs,
        num_tokens,
        glue_key,
        early_staged,
        attention_variant,
    ):
        # Unknown modes stay eager. Serving never captures a new metadata graph.
        if self._metadata_glue is not None and not self._metadata_glue.is_captured(
            glue_key
        ):
            return self._init_replay_attn_metadata_from_view(
                forward_batch,
                raw_bs=raw_bs,
                bs=bs,
                num_tokens=num_tokens,
                glue_key=None,
                attention_variant=attention_variant,
            )
        return super()._init_replay_attn_metadata(
            forward_batch,
            raw_bs=raw_bs,
            bs=bs,
            num_tokens=num_tokens,
            glue_key=glue_key,
            early_staged=early_staged,
            attention_variant=attention_variant,
        )

    def can_run_graph(self, forward_batch):
        if not qsa_context_supported(
            forward_batch.seq_lens_cpu, self._qsa_all_blocks_limit
        ):
            return False
        return super().can_run_graph(forward_batch)
