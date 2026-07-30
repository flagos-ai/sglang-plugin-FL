"""Patch FlashInfer attention backend classes for MetaX."""
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)


from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        # fast_decode_plan,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens


import importlib
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,          
    WrapperDispatch,                
    MultiItemScoringParams,         
    DecodeMetadata,                
    PrefillMetadata,              
    should_use_tensor_core,       
    global_workspace_buffer,       
    global_override_indptr_cpu,     
    FlashInferIndicesUpdaterPrefill,
    FlashInferIndicesUpdaterDecode, 
)



def patch_flashinfer_backend_classes() -> None:
    """Replace selected classes in sglang.srt.layers.attention.flashinfer_backend."""
    module_name = "sglang.srt.layers.attention.flashinfer_backend"

    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(f"failed to import {module_name}") from error

    class MetaXFlashInferAttnBackend(AttentionBackend):
        """MetaX replacement for FlashInferAttnBackend."""
        def __init__(
            self,
            model_runner: ModelRunner,
            skip_prefill: bool = False,
            kv_indptr_buf: Optional[torch.Tensor] = None,
            kv_last_page_len_buf: Optional[torch.Tensor] = None,
            init_new_workspace: bool = False,
        ):
            super().__init__()
            self.prefill_backend = "fa2"
            self.decode_backend = "fa2"

            # Store multi-item scoring flag for efficient access
            self.enable_mis = model_runner.server_args.enable_mis

            # FIXME: remove dllm workarounds from flashinfer
            self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
            self.is_dllm_model = self.dllm_config is not None

            # Parse constants
            self.decode_use_tensor_cores = should_use_tensor_core(
                kv_cache_dtype=model_runner.kv_cache_dtype,
                num_attention_heads=model_runner.model_config.num_attention_heads
                // get_attention_tp_size(),
                num_kv_heads=model_runner.model_config.get_num_kv_heads(
                    get_attention_tp_size()
                ),
            )
            self.max_context_len = model_runner.model_config.context_len
            self.skip_prefill = skip_prefill
            self.is_multimodal = model_runner.model_config.is_multimodal
            assert not (
                model_runner.sliding_window_size is not None
                and model_runner.model_config.is_encoder_decoder
            ), "Sliding window and cross attention are not supported together"

            if model_runner.sliding_window_size is not None:
                self.num_wrappers = 2
                self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
            elif model_runner.model_config.is_encoder_decoder:
                self.num_wrappers = 2
                self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
            else:
                self.num_wrappers = 1
                self.dispatch_reason = None

            # Qwen2/Qwen3 models require higher flashinfer workspace size
            if (
                "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
                or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
                or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
                or "Qwen3VLForConditionalGeneration"
                in model_runner.model_config.hf_config.architectures
                or "Qwen3VLMoeForConditionalGeneration"
                in model_runner.model_config.hf_config.architectures
                or "Qwen3_5MoeForConditionalGeneration"
                in model_runner.model_config.hf_config.architectures
                or "Qwen3_5ForConditionalGeneration"
                in model_runner.model_config.hf_config.architectures
            ):
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

            # When deterministic inference is enabled, tensor cores should be used for decode
            # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
            # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
            self.enable_deterministic = (
                model_runner.server_args.enable_deterministic_inference
            )
            self.prefill_split_tile_size = None
            self.decode_split_tile_size = None
            self.disable_cuda_graph_kv_split = False
            if self.enable_deterministic:
                self.decode_use_tensor_cores = True
                self.prefill_split_tile_size = get_int_env_var(
                    "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
                )
                self.decode_split_tile_size = get_int_env_var(
                    "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
                )
                self.disable_cuda_graph_kv_split = True
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

            self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

            # Allocate buffers
            global global_workspace_buffer
            if global_workspace_buffer is None:
                # different from flashinfer zero_init_global_workspace_buffer
                global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
                global_workspace_buffer = torch.empty(
                    global_workspace_size,
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            if init_new_workspace:
                self.workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            else:
                self.workspace_buffer = global_workspace_buffer
            max_bs = model_runner.req_to_token_pool.size
            if kv_indptr_buf is None:
                self.kv_indptr = [
                    torch.zeros(
                        (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                    )
                    for _ in range(self.num_wrappers)
                ]
            else:
                assert self.num_wrappers == 1
                self.kv_indptr = [kv_indptr_buf]

            if kv_last_page_len_buf is None:
                self.kv_last_page_len = torch.ones(
                    (max_bs,), dtype=torch.int32, device=model_runner.device
                )
            else:
                assert self.num_wrappers == 1
                self.kv_last_page_len = kv_last_page_len_buf

            if not self.skip_prefill:
                self.qo_indptr = [
                    torch.zeros(
                        (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                    )
                    for _ in range(self.num_wrappers)
                ]

            fmha_backend = "auto"
            if is_sm100_supported():
                # Disable CUTLASS backend when piecewise cuda graph is enabled
                # due to TMA descriptor initialization issues on B200
                if not model_runner.server_args.disable_piecewise_cuda_graph:
                    logger.warning(
                        "CUTLASS backend is disabled when piecewise cuda graph is enabled "
                        "due to TMA descriptor initialization issues on B200. "
                        "Using auto backend instead for stability."
                    )
                else:
                    fmha_backend = "cutlass"
            self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

            # Two wrappers: one for sliding window attention and one for full attention.
            # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
            self.prefill_wrappers_paged = []
            self.prefill_wrappers_verify = []
            self.decode_wrappers = []
            for _ in range(self.num_wrappers):
                if not skip_prefill:
                    self.prefill_wrappers_paged.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                        )
                    )
                    self.prefill_wrappers_verify.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                        )
                    )
                self.decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        # backend=self.decode_backend,
                        use_tensor_cores=self.decode_use_tensor_cores,
                    )
                )

            # Create indices updater
            if not skip_prefill:
                self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                    model_runner, self
                )  # for verify
            self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

            # Other metadata
            self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

            self.decode_cuda_graph_metadata = {}
            self.prefill_cuda_graph_metadata = {}  # For verify
            self.draft_extend_cuda_graph_metadata = {}  # For draft extend

        def _process_multi_item_scoring(
            self, forward_batch: ForwardBatch
        ) -> MultiItemScoringParams:
            """Process multi-item scoring tensors for FlashInfer attention.

            This method handles sequences containing multiple "items" separated by delimiter tokens,
            where each item needs specific attention patterns that respect item boundaries.

            The method produces four key tensors for FlashInfer:
            - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
            - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
            - token_pos_in_items_len: padding length for batch processing
            - max_item_len_ptr: uint16 tensor with max item length for each prompt

            Args:
                forward_batch: The forward batch containing input sequences and delimiter info

            Returns:
                MultiItemScoringParams: The processed multi-item scoring parameters

            Examples:
                Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
                token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

                Case 1: Single sequence
                Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
                Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
                Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
                - prefix_len_ptr: [7] (query length before first delimiter)
                - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
                - token_pos_in_items_len: 7 (actual length)
                - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

                Case 2: Batch processing (batch_size=2)
                Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
                Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
                After padding both to length 10:
                - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
                - token_pos_in_items_len: 10 (padded length for batch processing)
                - max_item_len_ptr: [2, 3] (max lengths per sequence)
            """

            if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
                return MultiItemScoringParams()

            precomputed_indices = forward_batch.multi_item_delimiter_indices
            if precomputed_indices is None:
                return MultiItemScoringParams()

            prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
            prefix_len_ptr, token_pos_in_items_ptr = [], []
            token_pos_in_items_len = 0
            device = forward_batch.input_ids.device

            # If no extend_seq_lens, treat whole batch as one sequence
            if extend_seq_lens is None or len(extend_seq_lens) <= 1:
                extend_seq_lens = [forward_batch.input_ids.size(0)]

            seq_start = 0
            for i, seq_len in enumerate(extend_seq_lens):
                seq_end = seq_start + seq_len
                delimiter_indices_cpu = precomputed_indices[i]
                if len(delimiter_indices_cpu) == 0:
                    seq_start = seq_end
                    continue

                first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
                delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
                prefix_len = first_delim + (
                    prefix_cache_lens[i] if prefix_cache_lens is not None else 0
                )
                prefix_len_ptr.append(prefix_len)

                # Compute relative positions within items using searchsorted (no GPU sync).
                #   suffix_range      = [0, 1, 2, 3, 4, ...]
                #   searchsorted      = bucket index for each position
                #   last_delim        = delimiter offset at start of current bucket
                #   pos_within_item   = suffix_range - last_delim
                suffix_len = seq_len - first_delim
                relative_positions = delimiter_indices - first_delim

                suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
                bucket_idx = torch.searchsorted(
                    relative_positions, suffix_range, right=True
                )
                last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
                pos_within_item = suffix_range - last_delim

                token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

                forward_batch.positions[seq_start + first_delim : seq_end] = (
                    prefix_len + pos_within_item - 1
                )

                seq_start = seq_end

            # Pad token_pos_in_items_ptr for batch processing
            if token_pos_in_items_ptr:
                token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
                token_pos_in_items_ptr = [
                    torch.cat(
                        [
                            t,
                            torch.zeros(
                                token_pos_in_items_len - t.numel(),
                                dtype=torch.uint16,
                                device=device,
                            ),
                        ]
                    )
                    for t in token_pos_in_items_ptr
                ]

            if not prefix_len_ptr or not token_pos_in_items_ptr:
                return MultiItemScoringParams()

            return MultiItemScoringParams(
                prefix_len_ptr=torch.tensor(
                    prefix_len_ptr, dtype=torch.uint32, device=device
                ),
                token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
                token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
                max_item_len_ptr=torch.stack(
                    [
                        t.to(torch.int32).max().to(torch.uint16)
                        for t in token_pos_in_items_ptr
                    ],
                    dim=0,
                ),
            )

        def init_forward_metadata(self, forward_batch: ForwardBatch):
            if forward_batch.forward_mode.is_decode_or_idle():
                self.indices_updater_decode.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    decode_wrappers=self.decode_wrappers,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                    fixed_split_size=self.decode_split_tile_size,
                    disable_split_kv=False,
                )
                self.forward_metadata = DecodeMetadata(self.decode_wrappers)
            elif forward_batch.forward_mode.is_draft_extend():
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_wrappers_paged,
                    use_ragged=False,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = PrefillMetadata(
                    self.prefill_wrappers_paged, False, False
                )
            elif forward_batch.forward_mode.is_target_verify():
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_wrappers_verify,
                    use_ragged=False,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = PrefillMetadata(
                    self.prefill_wrappers_verify, False, False
                )
            else:
                prefix_lens = forward_batch.extend_prefix_lens

                # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
                if self.is_multimodal or self.enable_mis:
                    # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                    # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                    #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                    # 2. Paged wrapper provides better control over attention masking needed
                    #    for respecting item boundaries in multi-item sequences
                    # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                    use_ragged = False
                    extend_no_prefix = False
                else:
                    use_ragged = (
                        not self.enable_deterministic
                        and not is_in_piecewise_cuda_graph()
                        and not self.use_paged
                    )
                    extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

                # Process multi-item scoring in attention backend instead of ForwardBatch
                multi_item_params = MultiItemScoringParams()
                if self.enable_mis:
                    # Use new backend-specific implementation
                    multi_item_params = self._process_multi_item_scoring(forward_batch)

                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefill_wrappers=self.prefill_wrappers_paged,
                    use_ragged=use_ragged,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                    fixed_split_size=self.prefill_split_tile_size,
                    multi_item_params=multi_item_params,
                    cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                )
                self.forward_metadata = PrefillMetadata(
                    self.prefill_wrappers_paged,
                    use_ragged,
                    extend_no_prefix,
                    multi_item_params,
                )

        def init_cuda_graph_state(
            self,
            max_bs: int,
            max_num_tokens: int,
            kv_indices_buf: Optional[torch.Tensor] = None,
        ):
            if kv_indices_buf is None:
                cuda_graph_kv_indices = torch.zeros(
                    (max_num_tokens * self.max_context_len,),
                    dtype=torch.int32,
                    device="cuda",
                )
            else:
                cuda_graph_kv_indices = kv_indices_buf

            self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
                cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
            ]

            # Ensure tensors are properly allocated
            for i in range(self.num_wrappers):
                # Force allocation by performing a small operation
                if len(self.cuda_graph_kv_indices[i]) > 0:
                    self.cuda_graph_kv_indices[i][0] = 0

            if not self.skip_prefill:
                self.cuda_graph_custom_mask = torch.zeros(
                    (max_num_tokens * self.max_context_len),
                    dtype=torch.uint8,
                    device="cuda",
                )
                self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
                self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

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
            if forward_mode.is_decode_or_idle():
                decode_wrappers = []
                for i in range(self.num_wrappers):
                    decode_wrappers.append(
                        BatchDecodeWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            # backend=self.decode_backend,
                            use_cuda_graph=True,
                            use_tensor_cores=self.decode_use_tensor_cores,
                            paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                            paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buffer=self.kv_last_page_len[
                                :num_tokens
                            ],
                        )
                    )
                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_decode.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens.cpu(),  # may add a little overhead in capture stage
                    seq_lens_sum,
                    decode_wrappers=decode_wrappers,
                    encoder_lens=encoder_lens,
                    spec_info=spec_info,
                    fixed_split_size=None,
                    disable_split_kv=self.disable_cuda_graph_kv_split,
                )
                self.decode_cuda_graph_metadata[bs] = decode_wrappers
                self.forward_metadata = DecodeMetadata(decode_wrappers)
                for i in range(self.num_wrappers):
                    decode_wrappers[i].begin_forward = partial(
                        fast_decode_plan, decode_wrappers[i]
                    )
            elif forward_mode.is_target_verify():
                # FlashInfer's prefill wrapper decides mask mode based on whether
                # `custom_mask_buf` is initialized (not whether a custom mask is provided).
                # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
                # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
                # FlashInfer will treat the (zero) buffer as a real mask and block attention.
                use_custom_mask = (
                    spec_info is not None
                    and getattr(spec_info, "custom_mask", None) is not None
                )
                prefill_wrappers = []
                for i in range(self.num_wrappers):
                    wrapper_kwargs = {}
                    if use_custom_mask:
                        wrapper_kwargs = {
                            "custom_mask_buf": self.cuda_graph_custom_mask,
                            "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                        }

                    prefill_wrappers.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            use_cuda_graph=True,
                            backend=self.prefill_backend,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                            **wrapper_kwargs,
                        )
                    )
                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_prefill.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens.cpu(),  # may add a little overhead in capture stage
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=prefill_wrappers,
                    use_ragged=False,
                    encoder_lens=encoder_lens,
                    spec_info=spec_info,
                )
                self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
                self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
            elif forward_mode.is_draft_extend():
                prefill_wrappers = []
                for i in range(self.num_wrappers):
                    prefill_wrappers.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                            use_cuda_graph=True,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        )
                    )

                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_prefill.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens.cpu(),  # may add a little overhead in capture stage
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=prefill_wrappers,
                    use_ragged=False,
                    encoder_lens=encoder_lens,
                    spec_info=spec_info,
                )
                self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
                self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
            elif forward_mode.is_dllm_extend():
                prefill_wrappers = []
                for i in range(self.num_wrappers):
                    prefill_wrappers.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=self.prefill_backend,
                            use_cuda_graph=True,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        )
                    )
                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_prefill.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens.cpu(),  # may add a little overhead in capture stage
                    seq_lens_sum,
                    prefix_lens=seq_lens - self.dllm_config.block_size,
                    prefill_wrappers=prefill_wrappers,
                    use_ragged=not self.use_paged,
                    encoder_lens=encoder_lens,
                    spec_info=None,
                )
                self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
                self.forward_metadata = PrefillMetadata(prefill_wrappers, True, False)
            else:
                raise ValueError(f"Invalid mode: {forward_mode=}")

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
            if forward_mode.is_decode_or_idle():
                self.indices_updater_decode.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    decode_wrappers=self.decode_cuda_graph_metadata[bs],
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                    fixed_split_size=None,
                    disable_split_kv=self.disable_cuda_graph_kv_split,
                )
            elif forward_mode.is_target_verify():
                self.indices_updater_prefill.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                    use_ragged=False,
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                )
            elif forward_mode.is_draft_extend():
                self.indices_updater_prefill.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    prefix_lens=None,
                    prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                    use_ragged=False,
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=spec_info,
                )
            elif forward_mode.is_dllm_extend():
                self.indices_updater_prefill.update(
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                    seq_lens_sum,
                    prefix_lens=seq_lens - self.dllm_config.block_size,
                    prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                    use_ragged=not self.use_paged,
                    encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                    spec_info=None,
                )
            else:
                raise ValueError("Invalid forward mode")

        def get_cuda_graph_seq_len_fill_value(self):
            return 1

        def update_verify_buffers_to_fill_after_draft(
            self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
        ):
            # Here is a no-op (same as Triton/AITER): this path reads the custom mask at
            # kernel-launch time rather than baking it into precomputed plan metadata, so
            # there is nothing to recompute.
            pass

        @debug_kernel_api
        def forward_extend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            layer: RadixAttention,
            forward_batch: ForwardBatch,
            save_kv_cache=True,
        ):
            prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
                self._get_wrapper_idx(layer)
            ]
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )

            logits_soft_cap = layer.logit_cap

            q = q.contiguous()
            if not self.forward_metadata.use_ragged:
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                        )

                causal = (
                    not layer.is_cross_attention
                    and layer.attn_type != AttentionType.ENCODER_ONLY
                )
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=causal,
                    sm_scale=layer.scaling,
                    # Disable sliding window attention for multi-item scoring:
                    # - Sliding window could cut across item boundaries, breaking semantic coherence
                    # - Multi-item sequences need full attention to properly handle delimiter tokens
                    # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                    #   provide more precise attention control than simple sliding windows
                    # - Item-aware masking takes precedence over window-based masking
                    window_left=(
                        layer.sliding_window_size
                        if not (
                            self.forward_metadata.multi_item_params
                            and self.forward_metadata.multi_item_params.is_enabled()
                        )
                        else -1
                    ),
                    logits_soft_cap=logits_soft_cap,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )
            else:
                # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
                # `forward_batch.token_to_kv_pool` for this layer. This enables attention over
                # previously cached context without re-materializing KV tensors (e.g., the
                # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
                if k is None and v is None:
                    k = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                    v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
                causal = True
                if (
                    layer.is_cross_attention
                    or layer.attn_type == AttentionType.ENCODER_ONLY
                ):
                    causal = False
                if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                    save_kv_cache = False

                if self.forward_metadata.extend_no_prefix:
                    # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                    # The FlashInfer head_dim limitation itself is tracked here:
                    # https://github.com/flashinfer-ai/flashinfer/issues/1048
                    o = self.prefill_wrapper_ragged.forward(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.head_dim),
                        causal=causal,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )

                else:
                    o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.head_dim),
                        causal=causal,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )
                    o2, s2 = prefill_wrapper_paged.forward_return_lse(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                        causal=False,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )

                    o, _ = merge_state(o1, s1, o2, s2)

                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        @debug_kernel_api
        def forward_decode(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            layer: RadixAttention,
            forward_batch: ForwardBatch,
            save_kv_cache=True,
        ):
            decode_wrapper = self.forward_metadata.decode_wrappers[
                self._get_wrapper_idx(layer)
            ]
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            # Call the wrapped function
            o = decode_wrapper.forward(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        def _get_wrapper_idx(self, layer: RadixAttention):
            if self.num_wrappers == 1:
                return 0

            if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
                return layer.sliding_window_size == -1
            if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
                return layer.is_cross_attention

            raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


    def metax_decode_call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
            # fixed_split_size=fixed_split_size,
            # disable_split_kv=(
            #     disable_split_kv if disable_split_kv is not None else False
            # ),
        )

        if locally_override:
            global_override_indptr_cpu = None

    def metax_prefill_call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            # fixed_split_size=fixed_split_size,
            # prefix_len_ptr=prefix_len_ptr,
            # token_pos_in_items_ptr=token_pos_in_items_ptr,
            # token_pos_in_items_len=token_pos_in_items_len,
            # max_item_len_ptr=max_item_len_ptr,
        )


    module.FlashInferAttnBackend = MetaXFlashInferAttnBackend
    module.FlashInferIndicesUpdaterPrefill.call_begin_forward = metax_prefill_call_begin_forward
    module.FlashInferIndicesUpdaterDecode.call_begin_forward = metax_decode_call_begin_forward
    logger.info(
        "patched flashinfer_backend classes: FlashInferAttnBackend, "
        "FlashInferIndicesUpdaterDecode"
    )

global_override_indptr_cpu = None


def fast_decode_plan(
    self,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    q_data_type: Optional[Union[str, torch.dtype]] = None,
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for FlashInferMultiStepDraftBackend.
    Modifications:
    - Remove unnecessary device-to-device copy for the cuda graph buffers.
    - Remove unnecessary host-to-device copy for the metadata buffers.
    """
    batch_size = len(last_page_len)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    # Handle data types consistently
    if data_type is not None:
        if q_data_type is None:
            q_data_type = data_type
        if kv_data_type is None:
            kv_data_type = data_type
    elif q_data_type is None:
        q_data_type = "float16"

    if kv_data_type is None:
        kv_data_type = q_data_type

    if self.use_tensor_cores:
        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        # Here we set fixed_split_size to -1 to avoid the assertion error in flashinfer's plan function
        if fixed_split_size is None:
            fixed_split_size = -1

    if self.is_cuda_graph_enabled:
        if batch_size != self._fixed_batch_size:
            raise ValueError(
                "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                " mismatches the batch size set during initialization {}".format(
                    batch_size, self._fixed_batch_size
                )
            )
        if len(indices) > len(self._paged_kv_indices_buf):
            raise ValueError(
                "The size of indices should be less than or equal to the allocated buffer"
            )
    else:
        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        self._paged_kv_last_page_len_buf = last_page_len
        if self.use_tensor_cores:
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

    # Create empty tensors for dtype info if needed
    empty_q_data = torch.empty(
        0,
        dtype=(
            getattr(torch, q_data_type) if isinstance(q_data_type, str) else q_data_type
        ),
        device=self.device,
    )

    empty_kv_cache = torch.empty(
        0,
        dtype=(
            getattr(torch, kv_data_type)
            if isinstance(kv_data_type, str)
            else kv_data_type
        ),
        device=self.device,
    )

    indptr_host = (
        global_override_indptr_cpu
        if global_override_indptr_cpu is not None
        else indptr.cpu()
    )

    with torch.cuda.device(self.device):

        if self.use_tensor_cores:
            # ALSO convert last_page_len to CPU
            if page_size == 1:
                # When page size is 1, last_page_len is always 1.
                # Directly construct the host tensor rather than executing a device-to-host copy.
                last_page_len_host = torch.ones(
                    (batch_size,), dtype=torch.int32, device="cpu"
                )
            else:
                last_page_len_host = last_page_len.cpu()

            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)

            try:
                # Make sure we pass exactly 15 arguments for tensor core version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    qo_indptr_host,
                    indptr_host,
                    kv_lens_arr_host,
                    batch_size,  # total_num_rows
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    head_dim,
                    head_dim,
                    False,  # causal
                    # window_left,
                    # fixed_split_size,
                    # disable_split_kv,
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}")
        else:
            try:
                # Make sure we pass exactly 15 arguments for standard version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    window_left,
                    logits_soft_cap,
                    head_dim,
                    head_dim,
                    empty_q_data,
                    empty_kv_cache,
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}")

    self._pos_encoding_mode = pos_encoding_mode
    self._window_left = window_left
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
