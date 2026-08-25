"""Unified strict Hygon FlashAttention backend for validated Qwen.

This standalone version exposes one FlashAttention implementation for both
eager execution and standard decode CUDA Graph. SGLang selects the execution
mode through ``disable_cuda_graph``; this backend has no eager/Graph backend
selector. It keeps ``TritonAttnBackend`` only for SGLang metadata and Graph
lifecycle compatibility. Attention prefill and decode never fall back to
Triton: unsupported configurations and request modes fail fast.

Lifecycle:
  - __init__ once per worker after model and memory-pool initialization
  - init_forward_metadata once per eager forward
  - init_cuda_graph_state once before standard graph capture
  - init_forward_metadata_*_cuda_graph during capture and replay
  - forward_extend / forward_decode per attention layer
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import (
    is_in_piecewise_cuda_graph,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

_PAGE_TABLE_DEBUG_ENV = "SGLANG_HCU_DEBUG_PAGE_TABLE"
_FLASH_ATTN_SUPPORTED_PAGE_SIZES = frozenset((64, 128, 256))
_FLASH_ATTN_PAGE_SIZE_DESCRIPTION = "64, 128, or 256"
_FLASH_ATTN_HEAD_DIM_DESCRIPTION = (
    "a multiple of 8 from 32 through 224, or exactly 256"
)


def _is_supported_flash_attn_head_dim(head_dim: int) -> bool:
    """Return whether the validated Hygon FlashAttention kernels support it."""
    return isinstance(head_dim, int) and head_dim % 8 == 0 and (
        32 <= head_dim <= 224 or head_dim == 256
    )


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"Invalid {name}={value!r}; expected a boolean value such as 0 or 1."
    )


def _validate_flash_attn_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    out_cache_loc: Optional[torch.Tensor] = None,
    require_disjoint_pages: bool = False,
):
    """Validate the SGLang token map against FlashAttention's block table.

    This helper intentionally synchronizes device tensors to the host. It is only
    called from the opt-in eager diagnostic path, never during Graph capture or
    replay.
    """
    if page_table.ndim != 2:
        raise RuntimeError(
            f"HCU FlashAttention debug expected a 2D page table, got "
            f"shape={tuple(page_table.shape)}."
        )

    batch_size = int(req_pool_indices.shape[0])
    if seq_lens.shape[0] < batch_size or page_table.shape[0] < batch_size:
        raise RuntimeError(
            "HCU FlashAttention debug metadata has inconsistent batch dimensions: "
            f"req_pool_indices={tuple(req_pool_indices.shape)}, "
            f"seq_lens={tuple(seq_lens.shape)}, page_table={tuple(page_table.shape)}."
        )
    if out_cache_loc is not None and out_cache_loc.shape[0] < batch_size:
        raise RuntimeError(
            "HCU FlashAttention debug out_cache_loc is smaller than the batch: "
            f"out_cache_loc={tuple(out_cache_loc.shape)}, batch_size={batch_size}."
        )

    req_indices_cpu = req_pool_indices[:batch_size].detach().to(
        device="cpu", dtype=torch.int64
    )
    seq_lens_cpu = seq_lens[:batch_size].detach().to(
        device="cpu", dtype=torch.int64
    )
    page_table_cpu = page_table[:batch_size].detach().to(
        device="cpu", dtype=torch.int64
    )
    out_cache_loc_cpu = (
        out_cache_loc[:batch_size].detach().to(device="cpu", dtype=torch.int64)
        if out_cache_loc is not None
        else None
    )

    page_owners = {}
    diagnostics = []
    for row in range(batch_size):
        req_idx = int(req_indices_cpu[row].item())
        seq_len = int(seq_lens_cpu[row].item())
        if req_idx < 0 or req_idx >= req_to_token.shape[0]:
            raise RuntimeError(
                f"HCU FlashAttention debug row={row} has invalid req_pool_index="
                f"{req_idx}; pool rows={req_to_token.shape[0]}."
            )
        if seq_len <= 0 or seq_len > req_to_token.shape[1]:
            raise RuntimeError(
                f"HCU FlashAttention debug row={row}, req={req_idx} has invalid "
                f"seq_len={seq_len}; token-map width={req_to_token.shape[1]}."
            )

        num_pages = (seq_len + page_size - 1) // page_size
        if num_pages > page_table_cpu.shape[1]:
            raise RuntimeError(
                f"HCU FlashAttention debug row={row}, req={req_idx}, "
                f"seq_len={seq_len} needs {num_pages} pages but the table has "
                f"{page_table_cpu.shape[1]} columns."
            )

        token_locs = req_to_token[req_idx, :seq_len].detach().to(
            device="cpu", dtype=torch.int64
        )
        expected_page_ids = []
        for logical_page in range(num_pages):
            begin = logical_page * page_size
            end = min(begin + page_size, seq_len)
            page_locs = token_locs[begin:end]
            page_id = int(torch.div(page_locs[0], page_size, rounding_mode="floor"))
            expected_locs = torch.arange(
                page_id * page_size,
                page_id * page_size + (end - begin),
                dtype=torch.int64,
            )
            if not torch.equal(page_locs, expected_locs):
                raise RuntimeError(
                    "HCU FlashAttention page allocation is not contiguous and "
                    f"{page_size}-aligned: row={row}, req={req_idx}, "
                    f"logical_page={logical_page}, page_id={page_id}, "
                    f"actual={page_locs.tolist()}, expected={expected_locs.tolist()}."
                )
            expected_page_ids.append(page_id)

            if require_disjoint_pages:
                owner = page_owners.setdefault(page_id, (row, req_idx, logical_page))
                if owner != (row, req_idx, logical_page):
                    raise RuntimeError(
                        "HCU FlashAttention active requests unexpectedly alias a KV "
                        f"page while radix cache is disabled: page_id={page_id}, "
                        f"first_owner={owner}, second_owner="
                        f"{(row, req_idx, logical_page)}."
                    )

        actual_page_ids = page_table_cpu[row, :num_pages].tolist()
        if actual_page_ids != expected_page_ids:
            raise RuntimeError(
                f"HCU FlashAttention page-table mismatch: row={row}, req={req_idx}, "
                f"seq_len={seq_len}, actual={actual_page_ids}, "
                f"expected={expected_page_ids}."
            )
        unused_page_ids = page_table_cpu[row, num_pages:].tolist()
        if any(page_id != -1 for page_id in unused_page_ids):
            raise RuntimeError(
                f"HCU FlashAttention page-table row contains stale entries: row={row}, "
                f"req={req_idx}, seq_len={seq_len}, unused={unused_page_ids}."
            )

        current_token_loc = int(token_locs[-1].item())
        if (
            out_cache_loc_cpu is not None
            and int(out_cache_loc_cpu[row].item()) != current_token_loc
        ):
            raise RuntimeError(
                f"HCU FlashAttention current-token mapping mismatch: row={row}, "
                f"req={req_idx}, seq_len={seq_len}, "
                f"out_cache_loc={int(out_cache_loc_cpu[row].item())}, "
                f"req_to_token={current_token_loc}."
            )

        diagnostics.append(
            {
                "row": row,
                "req_pool_index": req_idx,
                "seq_len": seq_len,
                "out_cache_loc": current_token_loc,
                "page_ids": expected_page_ids,
            }
        )
    return diagnostics


def _load_flash_attn_with_kvcache():
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "Strict HCU attention requires the vendor 'flash_attn' package, "
            "but it could not be imported."
        ) from exc

    kernel = getattr(flash_attn, "flash_attn_with_kvcache", None)
    if not callable(kernel):
        module_path = getattr(flash_attn, "__file__", "unknown")
        version = getattr(flash_attn, "__version__", "unknown")
        raise RuntimeError(
            "Strict HCU attention requires "
            "flash_attn.flash_attn_with_kvcache, but it is unavailable "
            f"(version={version}, path={module_path})."
        )
    return kernel


def _load_flash_attn_varlen_func():
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "Strict HCU attention requires the vendor 'flash_attn' package, "
            "but it could not be imported."
        ) from exc

    kernel = getattr(flash_attn, "flash_attn_varlen_func", None)
    if not callable(kernel):
        module_path = getattr(flash_attn, "__file__", "unknown")
        version = getattr(flash_attn, "__version__", "unknown")
        raise RuntimeError(
            "Strict HCU attention requires "
            "flash_attn.flash_attn_varlen_func, but it is unavailable "
            f"(version={version}, path={module_path})."
        )
    return kernel


class HCUAttnBackend(TritonAttnBackend):
    """FlashAttention-only Hygon backend for the controlled Qwen route."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # SGLang chooses eager versus standard decode Graph. Both modes call
        # the same strict FlashAttention forward implementation below.
        self._flash_attn_with_kvcache = None
        self._flash_attn_varlen_func = None

        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
        )

        self._flash_attn_input_dtype = model_runner.model_config.dtype
        self._flash_attn_kv_cache_dtype = model_runner.kv_cache_dtype
        self._flash_attn_page_size = model_runner.server_args.page_size
        self._flash_attn_head_dim = model_runner.model_config.head_dim
        self._flash_attn_v_head_dim = model_runner.model_config.v_head_dim

        self._flash_attn_eager_page_table = None
        self._flash_attn_cuda_graph_page_table = None
        self._flash_attn_forward_page_table = None
        self._flash_attn_forward_seq_lens = None
        self._flash_attn_prefill_cu_seqlens_q = None
        self._flash_attn_prefill_cu_seqlens_k = None
        self._flash_attn_prefill_max_seq_len = None
        self._flash_attn_prefill_max_k_seq_len = None
        self._flash_attn_prefill_page_table = None
        self._flash_attn_prefill_uses_paged_kv = False
        self._flash_attn_prefill_metadata_ready = False
        self._flash_attn_decode_logged = False
        self._flash_attn_prefill_logged = False
        self._flash_attn_prefill_paged_logged = False
        self._flash_attn_cuda_graph_replay_logged = False
        self._flash_attn_debug_page_table = _get_bool_env(_PAGE_TABLE_DEBUG_ENV)
        self._flash_attn_debug_require_disjoint_pages = bool(
            model_runner.server_args.disable_radix_cache
        )

        server_args = model_runner.server_args
        controlled_config_errors = []
        if not getattr(server_args, "disable_radix_cache", False):
            controlled_config_errors.append("disable_radix_cache must be True")
        if not getattr(server_args, "disable_piecewise_cuda_graph", False):
            controlled_config_errors.append(
                "disable_piecewise_cuda_graph must be True"
            )
        if getattr(server_args, "enable_breakable_cuda_graph", False):
            controlled_config_errors.append(
                "enable_breakable_cuda_graph must be False"
            )
        if getattr(server_args, "enable_deterministic_inference", False):
            controlled_config_errors.append(
                "enable_deterministic_inference must be False"
            )
        if controlled_config_errors:
            raise RuntimeError(
                "Strict HCU attention requires the controlled Qwen server "
                "configuration: " + "; ".join(controlled_config_errors) + "."
            )

        static_error = self._get_static_configuration_error()
        if static_error is not None:
            raise RuntimeError(
                "Strict HCU attention rejected the model configuration: "
                f"{static_error}. Triton attention fallback is disabled."
            )

        self._flash_attn_with_kvcache = _load_flash_attn_with_kvcache()
        self._flash_attn_varlen_func = _load_flash_attn_varlen_func()

        logger.info("HCU decode implementation: flash_attn (strict)")
        logger.info("HCU prefill implementation: flash_attn (strict)")
        if self._flash_attn_debug_page_table:
            logger.warning(
                "%s is enabled; eager decode will synchronize and validate "
                "FlashAttention page-table metadata",
                _PAGE_TABLE_DEBUG_ENV,
            )

    def _get_static_configuration_error(self) -> Optional[str]:
        # The strict backend rejects every unsupported model-wide condition.
        if self.use_mla:
            return "MLA is not supported"
        if self._flash_attn_page_size not in _FLASH_ATTN_SUPPORTED_PAGE_SIZES:
            return (
                f"page_size={self._flash_attn_page_size} is not supported; "
                "the validated Hygon path supports page_size="
                f"{_FLASH_ATTN_PAGE_SIZE_DESCRIPTION}"
            )
        if not _is_supported_flash_attn_head_dim(self._flash_attn_head_dim):
            return (
                f"head_dim={self._flash_attn_head_dim} is not supported; "
                "the validated Hygon path requires "
                f"{_FLASH_ATTN_HEAD_DIM_DESCRIPTION}"
            )
        if self._flash_attn_head_dim != self._flash_attn_v_head_dim:
            return "QK and V head dimensions differ"
        if self._flash_attn_kv_cache_dtype != torch.bfloat16:
            return (
                f"KV cache dtype {self._flash_attn_kv_cache_dtype} is not supported; "
                "the validated Hygon path requires bfloat16"
            )
        if self._flash_attn_input_dtype != torch.bfloat16:
            return (
                f"input dtype {self._flash_attn_input_dtype} is not supported; "
                "the validated Hygon path requires bfloat16"
            )
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            return "sliding-window attention is not supported"
        return None

    def _should_prepare_flash_attn_metadata(self, forward_mode, spec_info) -> bool:
        return forward_mode.is_decode_or_idle() and spec_info is None

    def _should_prepare_flash_attn_prefill_metadata(self, forward_batch) -> bool:
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        extend_lens = getattr(forward_batch, "extend_seq_lens", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        batch_size = forward_batch.batch_size
        if (
            prefix_lens is None
            or len(prefix_lens) != batch_size
            or extend_lens_cpu is None
            or len(extend_lens_cpu) != batch_size
            or not isinstance(extend_lens, torch.Tensor)
            or not isinstance(seq_lens, torch.Tensor)
            or not isinstance(seq_lens_cpu, torch.Tensor)
        ):
            return False

        expected_full_lens = [
            int(prefix_len) + int(extend_len)
            for prefix_len, extend_len in zip(prefix_lens, extend_lens_cpu)
        ]
        uses_paged_kv = any(int(prefix_len) > 0 for prefix_len in prefix_lens)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        token_map = getattr(self, "req_to_token", None)
        return (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and getattr(forward_batch, "global_forward_mode", None)
            in (None, ForwardMode.EXTEND)
            and forward_batch.spec_info is None
            and not is_in_piecewise_cuda_graph()
            and not is_in_breakable_cuda_graph()
            and all(int(prefix_len) >= 0 for prefix_len in prefix_lens)
            and all(int(extend_len) > 0 for extend_len in extend_lens_cpu)
            and extend_lens is not None
            and extend_lens.numel() >= batch_size
            and seq_lens is not None
            and seq_lens.numel() >= batch_size
            and seq_lens_cpu is not None
            and seq_lens_cpu.numel() >= batch_size
            and [
                int(length)
                for length in seq_lens_cpu[:batch_size].tolist()
            ]
            == expected_full_lens
            and (
                not uses_paged_kv
                or (
                    isinstance(req_pool_indices, torch.Tensor)
                    and req_pool_indices.numel() >= batch_size
                    and isinstance(token_map, torch.Tensor)
                    and token_map.ndim == 2
                    and max(expected_full_lens) <= token_map.shape[1]
                )
            )
        )

    def _prepare_flash_attn_prefill_metadata(self, forward_batch) -> None:
        batch_size = forward_batch.batch_size
        prefix_lens_cpu = [
            int(length) for length in forward_batch.extend_prefix_lens_cpu
        ]
        extend_lens_cpu = [
            int(length) for length in forward_batch.extend_seq_lens_cpu
        ]
        full_lens_cpu = [
            prefix_len + extend_len
            for prefix_len, extend_len in zip(prefix_lens_cpu, extend_lens_cpu)
        ]
        uses_paged_kv = any(prefix_len > 0 for prefix_len in prefix_lens_cpu)
        extend_lens = forward_batch.extend_seq_lens[:batch_size]
        cu_seqlens_q = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=extend_lens.device,
        )
        cu_seqlens_q[1:] = torch.cumsum(extend_lens, dim=0, dtype=torch.int32)

        page_table = None
        if uses_paged_kv:
            seq_lens = forward_batch.seq_lens[:batch_size]
            cu_seqlens_k = torch.zeros(
                batch_size + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            cu_seqlens_k[1:] = torch.cumsum(
                seq_lens,
                dim=0,
                dtype=torch.int32,
            )
            if self._flash_attn_page_size in _FLASH_ATTN_SUPPORTED_PAGE_SIZES:
                page_table = self._get_eager_page_table(
                    batch_size,
                    self.req_to_token.shape[1],
                )
                self._prepare_flash_attn_page_table(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    page_table,
                )
        else:
            cu_seqlens_k = cu_seqlens_q

        self._flash_attn_prefill_cu_seqlens_q = cu_seqlens_q
        self._flash_attn_prefill_cu_seqlens_k = cu_seqlens_k
        self._flash_attn_prefill_max_seq_len = max(extend_lens_cpu)
        self._flash_attn_prefill_max_k_seq_len = max(full_lens_cpu)
        self._flash_attn_prefill_page_table = page_table
        self._flash_attn_prefill_uses_paged_kv = uses_paged_kv
        self._flash_attn_prefill_metadata_ready = True

    def _get_eager_page_table(self, batch_size: int, max_seq_len: int):
        required_columns = max(
            1,
            (max_seq_len + self._flash_attn_page_size - 1)
            // self._flash_attn_page_size,
        )
        current = self._flash_attn_eager_page_table
        if (
            current is None
            or current.shape[0] < batch_size
            or current.shape[1] < required_columns
        ):
            rows = batch_size if current is None else max(batch_size, current.shape[0])
            columns = (
                required_columns
                if current is None
                else max(required_columns, current.shape[1])
            )
            current = torch.zeros(
                (rows, columns), dtype=torch.int32, device=self.device
            )
            self._flash_attn_eager_page_table = current
        return current[:batch_size]

    def _prepare_flash_attn_page_table(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if target.dtype != torch.int32 or target.ndim != 2:
            raise RuntimeError(
                "HCU FlashAttention page table must be a two-dimensional int32 "
                "tensor."
            )
        batch_size = target.shape[0]
        if batch_size == 0:
            return target

        target.fill_(-1)
        create_flashmla_kv_indices_triton[(batch_size,)](
            self.req_to_token,
            req_pool_indices[:batch_size],
            seq_lens[:batch_size],
            None,
            target,
            self.req_to_token.stride(0),
            target.stride(0),
            PAGED_SIZE=self._flash_attn_page_size,
        )
        return target

    @staticmethod
    def _max_seq_len_from_forward_batch(forward_batch) -> int:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            return int(seq_lens_cpu[: forward_batch.batch_size].max().item())
        return int(forward_batch.seq_lens[: forward_batch.batch_size].max().item())

    # ---- per-forward metadata ---------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        """Prepare SGLang base metadata and Hygon FlashAttention metadata."""
        # Triton metadata remains available for unsupported layers and modes.
        super().init_forward_metadata(forward_batch)
        self._flash_attn_forward_page_table = None
        self._flash_attn_forward_seq_lens = None
        self._flash_attn_prefill_cu_seqlens_q = None
        self._flash_attn_prefill_cu_seqlens_k = None
        self._flash_attn_prefill_max_seq_len = None
        self._flash_attn_prefill_max_k_seq_len = None
        self._flash_attn_prefill_page_table = None
        self._flash_attn_prefill_uses_paged_kv = False
        self._flash_attn_prefill_metadata_ready = False

        if self._should_prepare_flash_attn_metadata(
            forward_batch.forward_mode, forward_batch.spec_info
        ):
            batch_size = forward_batch.batch_size
            page_table = self._get_eager_page_table(
                batch_size, self._max_seq_len_from_forward_batch(forward_batch)
            )
            self._prepare_flash_attn_page_table(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                page_table,
            )
            self._flash_attn_forward_page_table = page_table
            # Eager ScheduleBatch sequence lengths are int64, while the Hygon
            # FlashAttention ABI requires int32. CUDA Graph buffers are already
            # int32 and are handled separately without changing their address.
            self._flash_attn_forward_seq_lens = forward_batch.seq_lens[
                :batch_size
            ].to(dtype=torch.int32)
            if getattr(self, "_flash_attn_debug_page_table", False):
                self._debug_validate_flash_attn_metadata(forward_batch, page_table)

        if self._should_prepare_flash_attn_prefill_metadata(forward_batch):
            self._prepare_flash_attn_prefill_metadata(forward_batch)

    def _debug_validate_flash_attn_metadata(
        self, forward_batch, page_table: torch.Tensor
    ) -> None:
        diagnostics = _validate_flash_attn_page_table(
            self.req_to_token,
            forward_batch.req_pool_indices[: forward_batch.batch_size],
            forward_batch.seq_lens[: forward_batch.batch_size],
            page_table,
            page_size=self._flash_attn_page_size,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            require_disjoint_pages=getattr(
                self, "_flash_attn_debug_require_disjoint_pages", False
            ),
        )
        for item in diagnostics:
            page_ids = item["page_ids"]
            if len(page_ids) > 16:
                page_ids = page_ids[:8] + ["..."] + page_ids[-8:]
            logger.info(
                "HCU FlashAttention page-table debug row=%d req=%d seq_len=%d "
                "out_cache_loc=%d page_ids=%s",
                item["row"],
                item["req_pool_index"],
                item["seq_len"],
                item["out_cache_loc"],
                page_ids,
            )

    # ---- CUDA / device graph hooks ----------------------------------------

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ) -> None:
        super().init_cuda_graph_state(
            max_bs,
            max_num_tokens,
            kv_indices_buf=kv_indices_buf,
            cuda_graph_num_kv_splits_buf=cuda_graph_num_kv_splits_buf,
        )

        self._flash_attn_forward_page_table = None
        self._flash_attn_forward_seq_lens = None

        max_pages_per_request = (
            self.max_context_len + self._flash_attn_page_size - 1
        ) // self._flash_attn_page_size
        self._flash_attn_cuda_graph_page_table = torch.zeros(
            (max_bs, max_pages_per_request),
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_flash_attn_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if self._flash_attn_cuda_graph_page_table is None:
            raise RuntimeError(
                "HCU FlashAttention CUDA Graph page-table buffer was not initialized."
            )
        if bs > self._flash_attn_cuda_graph_page_table.shape[0]:
            raise RuntimeError(
                "HCU FlashAttention CUDA Graph page-table buffer is too small: "
                f"required batch size={bs}, "
                f"capacity={self._flash_attn_cuda_graph_page_table.shape[0]}."
            )

        page_table = self._flash_attn_cuda_graph_page_table[:bs]
        self._prepare_flash_attn_page_table(
            req_pool_indices[:bs], seq_lens[:bs], page_table
        )
        self._flash_attn_forward_page_table = page_table
        self._flash_attn_forward_seq_lens = seq_lens[:bs]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self._flash_attn_forward_page_table = None
        self._flash_attn_forward_seq_lens = None
        if self._should_prepare_flash_attn_metadata(forward_mode, spec_info):
            self._prepare_flash_attn_cuda_graph_metadata(
                bs, req_pool_indices, seq_lens
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
    ) -> None:
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        self._flash_attn_forward_page_table = None
        self._flash_attn_forward_seq_lens = None
        if self._should_prepare_flash_attn_metadata(forward_mode, spec_info):
            self._prepare_flash_attn_cuda_graph_metadata(
                bs, req_pool_indices, seq_lens
            )
            if not getattr(self, "_flash_attn_cuda_graph_replay_logged", False):
                logger.info(
                    "HCU FlashAttention CUDA Graph replay metadata path is active"
                )
                self._flash_attn_cuda_graph_replay_logged = True

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Return the Triton-compatible padding value for graph sequence lengths."""
        return super().get_cuda_graph_seq_len_fill_value()

    def _get_prefill_error_reason(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache,
        sinks,
    ) -> Optional[str]:
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return "only ordinary EXTEND mode is supported"
        global_forward_mode = getattr(forward_batch, "global_forward_mode", None)
        if global_forward_mode not in (None, ForwardMode.EXTEND):
            return "the global forward mode is not ordinary EXTEND"
        if is_in_piecewise_cuda_graph():
            return "piecewise CUDA Graph prefill has not been validated"
        if is_in_breakable_cuda_graph():
            return "breakable CUDA Graph prefill has not been validated"
        if forward_batch.spec_info is not None:
            return "speculative prefill is not supported"
        if getattr(forward_batch, "attn_cp_metadata", None) is not None:
            return "context-parallel prefill is not supported"
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_lens is None:
            return "CPU prefix lengths are unavailable"
        if len(prefix_lens) != forward_batch.batch_size:
            return "prefix lengths do not match the batch size"
        if any(int(prefix_len) < 0 for prefix_len in prefix_lens):
            return "negative prefix lengths are not supported"
        uses_paged_kv = any(int(prefix_len) > 0 for prefix_len in prefix_lens)
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return "CPU extend lengths are unavailable"
        if len(extend_lens) != forward_batch.batch_size:
            return "extend lengths do not match the batch size"
        if any(int(extend_len) <= 0 for extend_len in extend_lens):
            return "non-positive extend lengths are not supported"
        extend_lens_device = getattr(forward_batch, "extend_seq_lens", None)
        if not isinstance(extend_lens_device, torch.Tensor):
            return "device extend lengths are unavailable"
        if extend_lens_device.numel() < forward_batch.batch_size:
            return "device extend lengths are smaller than the batch size"
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if not isinstance(seq_lens, torch.Tensor):
            return "full device sequence lengths are unavailable"
        if seq_lens.numel() < forward_batch.batch_size:
            return "full device sequence lengths are smaller than the batch size"
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if not isinstance(seq_lens_cpu, torch.Tensor):
            return "full CPU sequence lengths are unavailable"
        if seq_lens_cpu.numel() < forward_batch.batch_size:
            return "full CPU sequence lengths are smaller than the batch size"
        expected_full_lens = [
            int(prefix_len) + int(extend_len)
            for prefix_len, extend_len in zip(prefix_lens, extend_lens)
        ]
        actual_full_lens = [
            int(length)
            for length in seq_lens_cpu[: forward_batch.batch_size].tolist()
        ]
        if actual_full_lens != expected_full_lens:
            return "full sequence lengths do not match prefix plus extend lengths"
        if uses_paged_kv:
            if self._flash_attn_page_size not in _FLASH_ATTN_SUPPORTED_PAGE_SIZES:
                return (
                    "cached-prefix prefill requires page_size="
                    f"{_FLASH_ATTN_PAGE_SIZE_DESCRIPTION}"
                )
            if not save_kv_cache:
                return "cached-prefix prefill requires saving the new KV cache"
            req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
            if not isinstance(req_pool_indices, torch.Tensor):
                return "request-pool indices are unavailable"
            if req_pool_indices.numel() < forward_batch.batch_size:
                return "request-pool indices are smaller than the batch size"
            token_map = getattr(self, "req_to_token", None)
            if not isinstance(token_map, torch.Tensor) or token_map.ndim != 2:
                return "the request token map is unavailable"
            if max(expected_full_lens) > token_map.shape[1]:
                return "full sequence lengths exceed the request token-map width"
            if getattr(forward_batch, "attn_attend_prefix_cache", None) is not None:
                return "the chunked-prefix merge protocol is not supported"
            if getattr(forward_batch, "mha_one_shot", False):
                return "MHA one-shot prefill is not supported"
            if getattr(forward_batch, "mha_return_lse", False):
                return "prefill softmax-LSE output is not supported"
        if not getattr(self, "_flash_attn_prefill_metadata_ready", False):
            return "the current batch did not prepare FlashAttention prefill metadata"
        prepared_uses_paged_kv = getattr(
            self, "_flash_attn_prefill_uses_paged_kv", False
        )
        if bool(prepared_uses_paged_kv) != uses_paged_kv:
            return "prepared prefill metadata does not match the prefix state"
        if self.use_mla:
            return "MLA prefill is not supported"
        if self._flash_attn_input_dtype != torch.bfloat16:
            return "the validated Hygon prefill path requires bfloat16 inputs"
        if self._flash_attn_kv_cache_dtype != torch.bfloat16:
            return "the validated Hygon prefill path requires a bfloat16 KV cache"
        if not _is_supported_flash_attn_head_dim(
            self._flash_attn_head_dim
        ) or not _is_supported_flash_attn_head_dim(self._flash_attn_v_head_dim):
            return (
                "the validated Hygon prefill path requires head_dim to be "
                f"{_FLASH_ATTN_HEAD_DIM_DESCRIPTION}"
            )
        if getattr(self, "enable_deterministic", False):
            return "deterministic Hygon prefill has not been validated"
        if k is None or v is None:
            return "raw K and V tensors are required"
        if getattr(layer, "is_cross_attention", False):
            return "cross attention is not supported"
        if getattr(layer, "attn_type", AttentionType.DECODER) != AttentionType.DECODER:
            return "only causal decoder self-attention is supported"
        sliding_window_size = getattr(layer, "sliding_window_size", -1)
        if sliding_window_size is not None and sliding_window_size > -1:
            return "sliding-window attention is not supported"
        if layer.qk_head_dim != layer.v_head_dim:
            return "QK and V head dimensions differ"
        if layer.qk_head_dim != self._flash_attn_head_dim:
            return "the layer head dimension differs from the model head dimension"
        if layer.tp_q_head_num != self.num_head:
            return "the layer Q head count differs from the backend head count"
        if layer.tp_k_head_num != layer.tp_v_head_num:
            return "K and V head counts differ"
        if layer.tp_q_head_num % layer.tp_k_head_num != 0:
            return "the Q head count is not divisible by the KV head count"
        if layer.k_scale is not None or layer.v_scale is not None:
            return "quantized KV-cache scales are not supported"
        if sinks is not None:
            return "attention sinks are not supported"
        if getattr(layer, "xai_temperature_len", -1) > 0:
            return "XAI attention temperature is not supported"
        if (
            layer.logit_cap != 0
            and getattr(layer, "logit_capping_method", "tanh") != "tanh"
        ):
            return "the requested logit-capping method is not supported"
        if q.dtype != self._flash_attn_input_dtype:
            return f"query dtype {q.dtype} differs from the model input dtype"
        if (
            k.dtype != self._flash_attn_input_dtype
            or v.dtype != self._flash_attn_input_dtype
        ):
            return "K or V input dtype differs from the model input dtype"
        if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
            return "Q, K, and V do not use SGLang's expected prefill layout"
        num_tokens = sum(int(length) for length in extend_lens)
        if save_kv_cache:
            out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
            if out_cache_loc is None or out_cache_loc.numel() < num_tokens:
                return "output cache locations are smaller than the extend tokens"
        if q.shape != (num_tokens, layer.tp_q_head_num * layer.qk_head_dim):
            return "the flattened query shape does not match the extend metadata"
        if k.shape != (num_tokens, layer.tp_k_head_num, layer.qk_head_dim):
            return "the key shape does not match the extend metadata"
        if v.shape != (num_tokens, layer.tp_v_head_num, layer.v_head_dim):
            return "the value shape does not match the extend metadata"
        if uses_paged_kv:
            page_table_reason = self._get_paged_prefill_page_table_reason(
                forward_batch,
                q,
            )
            if page_table_reason is not None:
                return page_table_reason
            cache_layout_reason = self._get_paged_prefill_cache_layout_reason(
                layer,
                forward_batch,
                q,
            )
            if cache_layout_reason is not None:
                return cache_layout_reason
        return None

    def _get_paged_prefill_page_table_reason(
        self,
        forward_batch,
        q: torch.Tensor,
    ) -> Optional[str]:
        page_table = getattr(self, "_flash_attn_prefill_page_table", None)
        if page_table is None:
            return "cached-prefix prefill page-table metadata is unavailable"
        if page_table.dtype != torch.int32 or page_table.ndim != 2:
            return "cached-prefix prefill requires a two-dimensional int32 page table"
        if page_table.shape[0] != forward_batch.batch_size:
            return "cached-prefix prefill page-table rows do not match the batch"
        max_seq_len_k = getattr(
            self, "_flash_attn_prefill_max_k_seq_len", None
        )
        if max_seq_len_k is None or int(max_seq_len_k) <= 0:
            return "cached-prefix prefill maximum K sequence length is unavailable"
        required_columns = (
            int(max_seq_len_k) + self._flash_attn_page_size - 1
        ) // self._flash_attn_page_size
        if page_table.shape[1] < required_columns:
            return "cached-prefix prefill page-table width is too small"
        if page_table.device != q.device:
            return "cached-prefix prefill page table is on the wrong device"
        if not page_table.is_contiguous():
            return "cached-prefix prefill page table is not contiguous"
        return None

    def _get_paged_prefill_cache_layout_reason(
        self,
        layer,
        forward_batch,
        q: torch.Tensor,
    ) -> Optional[str]:
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        if token_to_kv_pool is None:
            return "the token-to-KV pool is unavailable"
        try:
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        except Exception as exc:
            return f"the token-to-KV pool cannot expose layer buffers: {exc}"
        if not isinstance(k_cache, torch.Tensor) or not isinstance(
            v_cache, torch.Tensor
        ):
            return "the token-to-KV pool did not return tensor buffers"
        if k_cache.ndim != 3 or v_cache.ndim != 3:
            return "cached-prefix KV buffers are not token-major 3D tensors"
        if k_cache.shape[0] != v_cache.shape[0]:
            return "cached-prefix K and V cache capacities differ"
        if k_cache.shape[0] % self._flash_attn_page_size != 0:
            return (
                "cached-prefix KV-cache capacity is not aligned to "
                f"page_size={self._flash_attn_page_size}"
            )
        if tuple(k_cache.shape[1:]) != (
            layer.tp_k_head_num,
            layer.qk_head_dim,
        ):
            return "cached-prefix K-cache shape does not match the layer"
        if tuple(v_cache.shape[1:]) != (
            layer.tp_v_head_num,
            layer.v_head_dim,
        ):
            return "cached-prefix V-cache shape does not match the layer"
        if k_cache.dtype != self._flash_attn_kv_cache_dtype:
            return "cached-prefix K-cache dtype does not match the backend"
        if v_cache.dtype != self._flash_attn_kv_cache_dtype:
            return "cached-prefix V-cache dtype does not match the backend"
        if k_cache.device != q.device or v_cache.device != q.device:
            return "cached-prefix KV buffers are on the wrong device"
        if not k_cache.is_contiguous() or not v_cache.is_contiguous():
            return "cached-prefix KV buffers are not contiguous"
        return None

    # ---- forward kernels --------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ordinary prefill/extend with strict Hygon FlashAttention."""
        error_reason = self._get_prefill_error_reason(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            sinks,
        )
        if error_reason is not None:
            raise RuntimeError(
                "Strict HCU FlashAttention prefill rejected this request: "
                f"{error_reason}. Triton attention fallback is disabled."
            )

        cu_seqlens_q = self._flash_attn_prefill_cu_seqlens_q
        cu_seqlens_k = self._flash_attn_prefill_cu_seqlens_k
        max_seq_len_q = self._flash_attn_prefill_max_seq_len
        max_seq_len_k = self._flash_attn_prefill_max_k_seq_len
        page_table = self._flash_attn_prefill_page_table
        uses_paged_kv = self._flash_attn_prefill_uses_paged_kv
        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seq_len_q is None
            or max_seq_len_k is None
            or cu_seqlens_q.dtype != torch.int32
            or cu_seqlens_k.dtype != torch.int32
            or cu_seqlens_q.ndim != 1
            or cu_seqlens_k.ndim != 1
            or cu_seqlens_q.numel() != forward_batch.batch_size + 1
            or cu_seqlens_k.numel() != forward_batch.batch_size + 1
            or cu_seqlens_q.device != q.device
            or cu_seqlens_k.device != q.device
        ):
            raise RuntimeError(
                "HCU FlashAttention prefill metadata was not prepared as int32 "
                "cumulative sequence lengths."
            )
        if self._flash_attn_varlen_func is None:
            raise RuntimeError("HCU FlashAttention prefill kernel was not loaded.")

        k_view = None
        v_view = None
        if uses_paged_kv:
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            k_view = k_cache.view(
                -1,
                self._flash_attn_page_size,
                layer.tp_k_head_num,
                layer.qk_head_dim,
            )
            v_view = v_cache.view(
                -1,
                self._flash_attn_page_size,
                layer.tp_v_head_num,
                layer.v_head_dim,
            )

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
            )

        num_tokens = q.shape[0]
        q_view = q.contiguous().view(
            num_tokens, layer.tp_q_head_num, layer.qk_head_dim
        )
        if uses_paged_kv:
            assert k_view is not None and v_view is not None
            if not getattr(self, "_flash_attn_prefill_paged_logged", False):
                logger.info(
                    "HCU flash_attn_varlen prefill path is active "
                    "(paged-prefix, page_size=%d, head_dim=%d)",
                    self._flash_attn_page_size,
                    layer.qk_head_dim,
                )
                self._flash_attn_prefill_paged_logged = True
        else:
            k_view = k.contiguous().view(
                num_tokens, layer.tp_k_head_num, layer.qk_head_dim
            )
            v_view = v.contiguous().view(
                num_tokens, layer.tp_v_head_num, layer.v_head_dim
            )
            if not getattr(self, "_flash_attn_prefill_logged", False):
                logger.info(
                    "HCU flash_attn_varlen prefill path is active "
                    "(no-prefix, head_dim=%d)",
                    layer.qk_head_dim,
                )
                self._flash_attn_prefill_logged = True

        kernel_kwargs = dict(
            q=q_view,
            k=k_view,
            v=v_view,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seq_len_q,
            max_seqlen_k=max_seq_len_k,
            dropout_p=0.0,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            deterministic=getattr(self, "enable_deterministic", False),
        )
        if uses_paged_kv:
            kernel_kwargs["block_table"] = page_table
        output = self._flash_attn_varlen_func(**kernel_kwargs)
        expected_shape = (
            num_tokens,
            layer.tp_q_head_num,
            layer.v_head_dim,
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "HCU flash_attn_varlen_func returned an unexpected output shape: "
                f"actual={tuple(output.shape)}, expected={expected_shape}."
            )
        if output.dtype != q.dtype or output.device != q.device:
            raise RuntimeError(
                "HCU flash_attn_varlen_func returned an unexpected output type: "
                f"actual=({output.dtype}, {output.device}), "
                f"expected=({q.dtype}, {q.device})."
            )
        return output.reshape(num_tokens, -1)

    def _get_runtime_error_reason(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache,
        sinks,
    ) -> Optional[str]:
        if (k is None) != (v is None):
            return "K and V must be provided together"
        if k is None and save_kv_cache:
            return "K and V are required when saving the KV cache"
        if not forward_batch.forward_mode.is_decode_or_idle():
            return "the forward mode is not normal decode"
        if forward_batch.spec_info is not None:
            return "speculative decoding is not supported"
        if getattr(layer, "is_cross_attention", False):
            return "cross attention is not supported"
        if getattr(layer, "attn_type", AttentionType.DECODER) != AttentionType.DECODER:
            return "only causal decoder self-attention is supported"
        sliding_window_size = getattr(layer, "sliding_window_size", -1)
        if sliding_window_size is not None and sliding_window_size > -1:
            return "sliding-window attention is not supported"
        if layer.qk_head_dim != layer.v_head_dim:
            return "QK and V head dimensions differ"
        if layer.qk_head_dim != self._flash_attn_head_dim:
            return "the layer head dimension differs from the model head dimension"
        if layer.tp_q_head_num != self.num_head:
            return "the layer Q head count differs from the backend head count"
        if layer.tp_k_head_num != layer.tp_v_head_num:
            return "K and V head counts differ"
        if layer.tp_q_head_num % layer.tp_k_head_num != 0:
            return "the Q head count is not divisible by the KV head count"
        if layer.k_scale is not None or layer.v_scale is not None:
            return "quantized KV-cache scales are not supported"
        if sinks is not None:
            return "attention sinks are not supported"
        if getattr(layer, "xai_temperature_len", -1) > 0:
            return "XAI attention temperature is not supported"
        if (
            layer.logit_cap != 0
            and getattr(layer, "logit_capping_method", "tanh") != "tanh"
        ):
            return "the requested logit-capping method is not supported"
        if q.dtype != self._flash_attn_input_dtype:
            return f"query dtype {q.dtype} differs from the model input dtype"
        if k is not None and (
            k.dtype != self._flash_attn_input_dtype
            or v.dtype != self._flash_attn_input_dtype
        ):
            return "K or V input dtype differs from the model input dtype"
        if q.shape[0] != forward_batch.batch_size:
            return "normal decode requires exactly one query token per request"
        return None

    def forward_decode(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run single-token decode with strict paged Hygon FlashAttention."""
        error_reason = self._get_runtime_error_reason(
            q, k, v, layer, forward_batch, save_kv_cache, sinks
        )
        if error_reason is not None:
            raise RuntimeError(
                "Strict HCU FlashAttention decode rejected this request: "
                f"{error_reason}. Triton attention fallback is disabled."
            )

        page_table = self._flash_attn_forward_page_table
        seq_lens = self._flash_attn_forward_seq_lens
        if page_table is None or page_table.dtype != torch.int32:
            raise RuntimeError(
                "HCU FlashAttention decode requires a prepared int32 page table; "
                "metadata initialization did not provide one."
            )
        if seq_lens is None or seq_lens.dtype != torch.int32:
            raise RuntimeError(
                "HCU FlashAttention decode requires prepared int32 sequence lengths; "
                "metadata initialization did not provide them."
            )

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
            )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        if (
            k_cache.shape[0] % self._flash_attn_page_size != 0
            or v_cache.shape[0] % self._flash_attn_page_size != 0
        ):
            raise RuntimeError(
                "HCU FlashAttention KV-cache capacity is not page-aligned to "
                f"page_size={self._flash_attn_page_size}."
            )

        batch_size = forward_batch.batch_size
        q = q.reshape(
            batch_size,
            1,
            layer.tp_q_head_num,
            layer.qk_head_dim,
        )
        k_cache = k_cache.view(
            -1,
            self._flash_attn_page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = v_cache.view(
            -1,
            self._flash_attn_page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )

        if not self._flash_attn_decode_logged:
            logger.info(
                "HCU flash_attn_with_kvcache decode path is active "
                "(page_size=%d, num_splits=1)",
                self._flash_attn_page_size,
            )
            self._flash_attn_decode_logged = True

        output = self._flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=seq_lens,
            block_table=page_table,
            softmax_scale=layer.scaling,
            causal=True,
            softcap=layer.logit_cap,
            num_splits=1,
        )
        return output.reshape(batch_size, -1)
