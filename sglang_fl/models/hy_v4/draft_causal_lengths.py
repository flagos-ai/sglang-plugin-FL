# SPDX-License-Identifier: Apache-2.0
"""HY4 draft-decode causal metadata; no attention/indexer math.

Native Eagle supplies the committed prefix L and writes draft step i to L+i.
Each child therefore reads L+i+1 keys. Preserve native construction/copy APIs
for every derived metadata field, including page tables and clipped TopK.
"""

from contextvars import ContextVar
from copy import copy
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace


def causal_backend_type(native_base, build_inner_fb_view, forward_batch_type):
    """The dependency seam also permits a CPU integer-only contract test."""

    class HYV4CausalDraftMetadata(native_base):
        def __init__(self, original, model_runner):
            self.__dict__.update(original.__dict__)
            self._hy4_device = model_runner.device
            self._hy4_capacity = 0
            self._hy4_storage_pinned = False
            self._hy4_active_rows_host = 0
            self._hy4_views = {}

        def _reserve(self, rows):
            if rows <= self._hy4_capacity:
                return
            if self._hy4_storage_pinned:
                raise RuntimeError(
                    "causal draft metadata cannot resize captured storage"
                )
            import torch

            count = len(self.attn_backends)
            self._hy4_seq = torch.empty(
                (count, rows), dtype=torch.int64, device=self._hy4_device
            )
            self._hy4_seq_cpu = torch.empty(
                (count, rows), dtype=torch.int64, device="cpu"
            )
            # One stable step-major owner shared by both the INT8 indexer
            # writer and native MLA writer. Each step row has stride one.
            self._hy4_cache_locs = torch.empty(
                (count + 1, rows), dtype=torch.int64, device=self._hy4_device
            )
            self._hy4_row_ids = torch.arange(
                rows, dtype=torch.int32, device=self._hy4_device
            )
            self._hy4_inactive = torch.empty(
                rows, dtype=torch.bool, device=self._hy4_device
            )
            self._hy4_active_rows = torch.zeros(
                (), dtype=torch.int32, device=self._hy4_device
            )
            self._hy4_capacity = rows

        def stage_per_step_cache_locs(self, locations):
            if locations.ndim != 2 or locations.shape[0] != len(self.attn_backends) + 1:
                raise ValueError("unexpected HY4 per-step cache-location shape")
            # A single element is safe for a raw-pointer reader whatever its
            # nominal stride. Preserve the already-qualified q1 value path.
            if locations.shape[1] <= 1 or locations.stride(1) == 1:
                return locations
            self._reserve(locations.shape[1])
            out = self._hy4_cache_locs[:, : locations.shape[1]]
            if locations.dtype != out.dtype or locations.device != out.device:
                raise TypeError("unexpected HY4 cache-location dtype/device")
            out.copy_(locations)
            return out

        def stage_active_rows(self, raw_rows):
            if not 0 <= raw_rows <= self._hy4_capacity:
                raise ValueError("active rows exceed causal draft storage")
            self._hy4_active_rows_host = raw_rows
            self._hy4_active_rows.fill_(raw_rows)

        def init_cuda_graph_state(self, max_bs, max_num_tokens):
            # Exactly the original children and native metadata storage.
            super().init_cuda_graph_state(max_bs, max_num_tokens)
            self._reserve(max_bs)
            self._hy4_storage_pinned = True
            self.stage_active_rows(0)  # Cold capture contains only native dummy rows.

        def _step_view(self, fb, step):
            import torch

            rows = fb.batch_size
            self._reserve(rows)
            if fb.seq_lens.numel() != rows:
                raise ValueError("causal draft requires one sequence length per row")
            offset = step + 1
            shifted = self._hy4_seq[step, :rows]
            torch.add(fb.seq_lens, offset, out=shifted)
            torch.ge(
                self._hy4_row_ids[:rows],
                self._hy4_active_rows,
                out=self._hy4_inactive[:rows],
            )
            shifted.masked_fill_(
                self._hy4_inactive[:rows],
                self.attn_backends[step].get_cuda_graph_seq_len_fill_value(),
            )
            cpu = None
            active = min(rows, self._hy4_active_rows_host)
            if fb.seq_lens_cpu is not None:
                cpu = self._hy4_seq_cpu[step, :rows]
                torch.add(fb.seq_lens_cpu, offset, out=cpu)
                cpu[active:].fill_(
                    self.attn_backends[step].get_cuda_graph_seq_len_fill_value()
                )
            total = (
                None if fb.seq_lens_sum is None else fb.seq_lens_sum + active * offset
            )
            updates = {"seq_lens": shifted, "seq_lens_cpu": cpu, "seq_lens_sum": total}
            if type(fb) is SimpleNamespace:
                # Native build_inner_fb_view returns this exact type. copy()
                # gives an independent namespace dictionary, retaining every
                # untouched native owner without mutating the base view.
                view = copy(fb)
                view.__dict__.update(updates)
                return view
            if type(fb) is forward_batch_type:
                return replace(fb, **updates)
            raise TypeError("unsupported causal draft ForwardBatch view type")

        def init_forward_metadata(self, forward_batch):
            self._reserve(forward_batch.batch_size)
            self.stage_active_rows(forward_batch.batch_size)
            for step, child in enumerate(self.attn_backends):
                child.init_forward_metadata(self._step_view(forward_batch, step))

        def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
            self._reserve(forward_batch.batch_size)
            base = forward_batch
            if in_capture:
                # Same native inner decode view as the unmodified container.
                from sglang.srt.model_executor.forward_batch_info import ForwardMode

                base = build_inner_fb_view(
                    forward_batch,
                    bs=forward_batch.batch_size,
                    forward_mode=ForwardMode.DECODE,
                )
                self.stage_active_rows(0)
            views = []
            for step, child in enumerate(self.attn_backends):
                view = self._step_view(base, step)
                child.init_forward_metadata_out_graph(view, in_capture=in_capture)
                views.append(view)
            self._hy4_views[base.batch_size] = views

        def init_forward_metadata_in_graph(self, forward_batch):
            # Native 0.5.18 DSA children have an empty in-graph method. Keep the
            # same lifecycle, without recording another update against base L.
            views = self._hy4_views[forward_batch.batch_size]
            for child, view in zip(self.attn_backends, views):
                child.init_forward_metadata_in_graph(view)

    return HYV4CausalDraftMetadata


def prepare_patches(previous_factory):
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnMultiStepBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        build_inner_fb_view,
    )
    from sglang.srt.speculative import eagle_worker_v2
    from sglang.srt.speculative.draft_utils import DraftBackendFactory
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )

    from .attention_backend import HYV4SparseAttnBackend
    from .register import _hy4

    Fixed = causal_backend_type(
        DeepseekSparseAttnMultiStepBackend, build_inner_fb_view, ForwardBatch
    )
    previous_execute = EAGLEDraftCudaGraphRunner.execute
    previous_draft_forward = eagle_worker_v2.EagleDraftWorker.draft_forward
    previous_locations = eagle_worker_v2.per_step_draft_out_cache_loc
    active_container = ContextVar("hy4_causal_draft_cache_locations", default=None)

    @wraps(previous_factory)
    def create_decode(factory):
        result = previous_factory(factory)
        runner = factory.draft_model_runner
        if (
            not _hy4(runner.model_config.hf_config)
            or getattr(runner, "is_draft_worker", None) is not True
            or getattr(getattr(runner, "ps", None), "attn_cp_size", None) != 1
            or getattr(getattr(runner, "ps", None), "attn_dcp_size", None) != 1
            or factory.topk != 1
            or factory.speculative_num_steps < 2
        ):
            return result
        name, original = result
        if not all(
            isinstance(child, HYV4SparseAttnBackend) for child in original.attn_backends
        ):
            raise RuntimeError(
                "HY4 causal draft metadata requires the sink-aware draft factory first"
            )
        return name, Fixed(original, runner)

    @wraps(previous_execute)
    def execute(graph, forward_batch, *args, **kwargs):
        container = graph.draft_attn_backend
        if isinstance(container, Fixed):
            # Stage the real count BEFORE native execute replaces it by its
            # graph bucket. This is host-known metadata, never a tensor read.
            container.stage_active_rows(forward_batch.batch_size)
        return previous_execute(graph, forward_batch, *args, **kwargs)

    @wraps(previous_draft_forward)
    def draft_forward(worker, forward_batch, *args, **kwargs):
        container = getattr(worker, "draft_attn_backend", None)
        token = active_container.set(
            container if isinstance(container, Fixed) else None
        )
        try:
            return previous_draft_forward(worker, forward_batch, *args, **kwargs)
        finally:
            active_container.reset(token)

    @wraps(previous_locations)
    def per_step_locations(*args, **kwargs):
        locations = previous_locations(*args, **kwargs)
        container = active_container.get()
        return (
            locations
            if container is None
            else container.stage_per_step_cache_locs(locations)
        )

    return [
        (
            DraftBackendFactory,
            "_create_dsa_decode_backend",
            previous_factory,
            create_decode,
        ),
        (EAGLEDraftCudaGraphRunner, "execute", previous_execute, execute),
        (
            eagle_worker_v2.EagleDraftWorker,
            "draft_forward",
            previous_draft_forward,
            draft_forward,
        ),
        (
            eagle_worker_v2,
            "per_step_draft_out_cache_loc",
            previous_locations,
            per_step_locations,
        ),
    ]
