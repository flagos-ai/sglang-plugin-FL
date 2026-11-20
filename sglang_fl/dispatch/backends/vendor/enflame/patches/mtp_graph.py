"""GCU full-graph capture for target, EAGLE draft and draft-extend."""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)


class _GraphDeviceModule:
    """Translate PyTorch's CUDA graph keyword to the GCU positional API."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def graph(self, cuda_graph, **kwargs):
        return self._module.graph(cuda_graph, **kwargs)


def capture_one_gcu(original, self, *args, **kwargs):
    if torch.device(self._cuda_graph_runner.device).type != "gcu":
        return original(self, *args, **kwargs)
    module = self._device_module
    # The proxy is confined to this backend instance and capture call.
    self._device_module = _GraphDeviceModule(module)
    try:
        return original(self, *args, **kwargs)
    finally:
        self._device_module = module


def capture_draft_graphs_gcu(original, self):
    if torch.device(self.device).type != "gcu":
        return original(self)

    from sglang.srt.speculative import eagle_worker_v2 as eagle

    self.cuda_graph_runner = None
    self.cuda_graph_runner_for_draft_extend = None
    if eagle.check_cuda_graph_backend(eagle.Phase.DECODE, eagle.Backend.DISABLED):
        return
    if eagle.get_model().model_impl == "mindspore":
        return

    backend = eagle.get_exec().graph.cuda_graph_config.decode.backend
    capture_bs, _ = eagle.get_batch_sizes_to_capture(self.draft_runner)

    def capture(attr, kind, runner_cls, width):
        start = time.perf_counter()
        before = eagle.get_available_gpu_memory(self.device, self.gpu_id)
        eagle.log_info_on_rank0(
            logger,
            f"Capture GCU {kind} graph begin. backend={backend}, "
            f"num_tokens_per_req={width}, bs={capture_bs}, avail mem={before:.2f} GB",
        )
        setattr(self, attr, runner_cls(self))
        after = eagle.get_available_gpu_memory(self.device, self.gpu_id)
        elapsed = time.perf_counter() - start
        self._specialized_graph_memory_usage[kind] = (
            self._specialized_graph_memory_usage.get(kind, 0.0) + before - after
        )
        self._specialized_graph_time_usage[kind] = (
            self._specialized_graph_time_usage.get(kind, 0.0) + elapsed
        )
        eagle.log_info_on_rank0(
            logger,
            f"Capture GCU {kind} graph end. elapsed={elapsed:.2f} s, "
            f"mem usage={before - after:.2f} GB, avail mem={after:.2f} GB",
        )

    if self.speculative_num_steps > 1:
        capture(
            "cuda_graph_runner", "draft_decode",
            eagle.EAGLEDraftCudaGraphRunner, self.topk,
        )
    if (
        self.draft_extend_attn_backend is not None
        and not eagle.envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
    ):
        capture(
            "cuda_graph_runner_for_draft_extend", "draft_extend",
            eagle.EAGLEDraftExtendCudaGraphRunner, self.speculative_num_draft_tokens,
        )


def _select_topk1_later(i, topk_p, topk_index, hidden_states, scores):
    # With one candidate, top-k and gather are identity selections. Keeping the
    # chain on device avoids FlagGems gather's host-side bounds check in capture.
    expanded = scores.unsqueeze(2) * topk_p.view(-1, 1, 1)
    indices = topk_index.view(-1, 1)
    parents = torch.full_like(indices, i)
    return (
        indices.flatten(), hidden_states, expanded.flatten(start_dim=1),
        (expanded, indices, parents),
    )


def select_tokens_later_gcu(original, i, topk_p, topk_index, hidden_states, scores, topk):
    if topk != 1 or topk_p.device.type != "gcu":
        return original(i, topk_p, topk_index, hidden_states, scores, topk)
    return _select_topk1_later(i, topk_p, topk_index, hidden_states, scores)


def _draft_extend_metadata(
    cache_seqlens_int32, cu_seqlens_k, cu_seqlens_q, page_table,
    req_to_token, req_pool_indices, seq_lens, extend_seq_lens, page_size,
    full_to_swa_index_mapping=None, swa_page_table=None,
    out_cache_loc=None, swa_out_cache_loc=None,
):
    cache_seqlens_int32.copy_(seq_lens)
    cu_seqlens_k[:1].zero_()
    cu_seqlens_q[:1].zero_()
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, 0, dtype=torch.int32))
    cu_seqlens_q[1:].copy_(torch.cumsum(extend_seq_lens, 0, dtype=torch.int32))
    columns = torch.arange(page_table.shape[1], device=seq_lens.device) * page_size
    page_indices = req_to_token[req_pool_indices[:, None], columns[None, :]]
    live = columns[None, :] < seq_lens[:, None]
    page_table.copy_(torch.where(live, page_indices // page_size, page_table))
    if full_to_swa_index_mapping is not None:
        # Mask before the lookup: padded req_to_token entries are not live ids.
        safe_indices = torch.where(live, page_indices, 0)
        swa_pages = full_to_swa_index_mapping[safe_indices] // page_size
        swa_page_table.copy_(torch.where(live, swa_pages, swa_page_table))
        swa_out_cache_loc.copy_(full_to_swa_index_mapping[out_cache_loc])


def draft_extend_metadata_gcu(original, cache_seqlens_int32, *args, **kwargs):
    if cache_seqlens_int32.device.type != "gcu":
        return original(cache_seqlens_int32, *args, **kwargs)
    return _draft_extend_metadata(cache_seqlens_int32, *args, **kwargs)


def patch_mtp_graph():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_one",
        capture_one_gcu,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._capture_cuda_graphs",
        capture_draft_graphs_gcu,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.spec_utils._select_top_k_tokens_later",
        select_tokens_later_gcu,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.metadata.draft_extend_set_metadata",
        draft_extend_metadata_gcu,
        HookType.AROUND,
    )
