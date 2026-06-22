"""Template: skeleton AttentionBackend subclass for a new OOT vendor.

Copy this file (along with the rest of ``vendor/template/``) into
``vendor/<your_vendor>/`` and replace the ``NotImplementedError`` stubs with
real implementations on top of your vendor SDK.

Lifecycle (each method is called by sglang's ModelRunner at a specific stage):
  - ``__init__``                      once per worker, after model load
  - ``init_cuda_graph_state``         once, before graph capture
  - ``init_forward_metadata_*``       at every graph capture/replay
  - ``init_forward_metadata``         at every forward pass
  - ``forward_extend`` / ``forward_decode``  per layer per token batch

You only have to override what your hardware needs. Methods you leave
unimplemented stay at the base class's ``NotImplementedError`` — sglang will
surface a clear error if it ever tries to call one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


class TemplateOOTAttnBackend(AttentionBackend):
    """Replace with ``<YourVendor>OOTAttnBackend``."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        # TODO: cache anything you need from model_runner.model_config,
        # model_runner.server_args, model_runner.kv_cache_dtype, etc.
        # Don't hold a reference to model_runner itself across forwards —
        # it owns the GPU memory pool and you don't want a cycle.
        raise NotImplementedError("TemplateOOTAttnBackend.__init__")

    # ---- per-forward metadata ------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Prepare any per-batch state (kv indices, block tables, masks, ...).

        Called once per forward at the top of ModelRunner.forward(). Stash
        results on ``self`` so forward_extend / forward_decode can read them
        without recomputing.
        """
        raise NotImplementedError("TemplateOOTAttnBackend.init_forward_metadata")

    # ---- forward kernels -----------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Attention kernel for prefill / extend.

        - q/k/v already have layer-local shapes (see RadixAttention).
        - If save_kv_cache is True, write new k/v into ``layer.token_to_kv_pool``.
        - Return the attention output tensor with shape (num_tokens, hidden).
        """
        raise NotImplementedError("TemplateOOTAttnBackend.forward_extend")

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Attention kernel for single-token decode."""
        raise NotImplementedError("TemplateOOTAttnBackend.forward_decode")

    # ---- CUDA / device graph hooks (override only if you support graphs) ----

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Allocate persistent workspace for graph capture.

        Skip this override if your backend doesn't support device graphs;
        ``PlatformFL.support_cuda_graph()`` controls whether sglang attempts
        to call graph-related methods at all.
        """
        raise NotImplementedError("TemplateOOTAttnBackend.init_cuda_graph_state")

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        raise NotImplementedError(
            "TemplateOOTAttnBackend.init_forward_metadata_capture_cuda_graph"
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        raise NotImplementedError(
            "TemplateOOTAttnBackend.init_forward_metadata_replay_cuda_graph"
        )

    def get_cuda_graph_seq_len_fill_value(self):
        """0 for most kernels; some need 1 to avoid divide-by-zero in masks."""
        return 0
