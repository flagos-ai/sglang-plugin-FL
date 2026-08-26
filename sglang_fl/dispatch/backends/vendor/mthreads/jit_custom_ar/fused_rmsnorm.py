# SPDX-License-Identifier: Apache-2.0

import os
from functools import lru_cache
from typing import Any, Optional

import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_weak_contiguous,
)


@lru_cache(maxsize=None)
def _env_flag(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in ("1", "true", "yes", "on")
    return default


def _fused_rmsnorm_multi_rank_enabled() -> bool:
    return _env_flag(
        (
            "SGLANG_CUSTOM_AR_FUSED_RMSNORM_MULTI_RANK",
            "SGL_CUSTOM_AR_FUSED_RMSNORM_MULTI_RANK",
        ),
        True,
    )


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _fused_rmsnorm_tp2_single_kernel_shape(rows: int, hidden: int) -> bool:
    if not (1 <= hidden <= 16384):
        return False
    if hidden <= 8192 and _is_power_of_2(hidden):
        return True
    if hidden % 8 != 0:
        return rows <= 8192
    return rows <= 512


def _fused_rmsnorm_tp2_row_one_shape(rows: int, hidden: int) -> bool:
    if not (1 <= rows <= 131072):
        return False
    # The row-one kernel assigns one vec8 chunk to each thread. Keep the
    # Python dispatch aligned with the kernel's 1024-thread launch limit.
    return hidden % 8 == 0 and 0 < hidden <= 8 * 1024


def _fused_rmsnorm_tp2_default_shape(rows: int, hidden: int) -> bool:
    if not (1 <= rows <= 131072):
        return False
    if hidden % 8 != 0:
        return False
    if 0 < hidden <= 16384:
        return True
    return _fused_rmsnorm_tp2_single_kernel_shape(
        rows, hidden
    ) or _fused_rmsnorm_tp2_row_one_shape(rows, hidden)


def _fused_rmsnorm_tp8_row_one_shape(rows: int, hidden: int) -> bool:
    if not (1 <= rows <= 131072):
        return False
    return hidden % 8 == 0 and 0 < hidden <= 8 * 1024


def _fused_rmsnorm_tp8_shape(rows: int, hidden: int) -> bool:
    return hidden % 8 == 0 and 0 < hidden <= 16384 and 1 <= rows <= 131072


def _fused_rmsnorm_tp4_row_one_shape(rows: int, hidden: int) -> bool:
    if not (1 <= rows <= 131072):
        return False
    return hidden % 8 == 0 and 0 < hidden <= 8 * 1024


def _fused_rmsnorm_tp4_shape(rows: int, hidden: int) -> bool:
    return hidden % 8 == 0 and 0 < hidden <= 16384 and 1 <= rows <= 131072


def _fused_rmsnorm_default_shape(world_size: int, rows: int, hidden: int) -> bool:
    if rows < 1:
        return False
    if world_size == 2:
        return _fused_rmsnorm_tp2_default_shape(rows, hidden)
    if world_size == 4:
        return _fused_rmsnorm_tp4_shape(rows, hidden)
    if world_size == 8:
        return _fused_rmsnorm_tp8_shape(rows, hidden)
    return False


class MusaJitCustomAllreduceRMSNorm:
    def __init__(self, comm: Any) -> None:
        self._comm = comm
        self._fused_rmsnorm_unregistered_launchers = {}
        self._fused_rmsnorm_registered_launchers = {}
        self._fused_rmsnorm_row_registered_launchers = {}
        self._fused_rmsnorm_row_unregistered_launchers = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._comm, name)

    def _should_fused_rmsnorm_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16):
            return False
        if inp.numel() * inp.element_size() > self.max_size:
            return False
        return is_weak_contiguous(inp)

    def fused_allreduce_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.world_size not in (2, 4, 8):
            return None
        if self.world_size != 2 and not _fused_rmsnorm_multi_rank_enabled():
            return None
        if input_.dtype not in (torch.float16, torch.bfloat16):
            return None
        if input_.dim() != 2:
            return None
        if residual_inp_.shape != input_.shape:
            return None
        if weight_.dim() != 1 or weight_.numel() != input_.shape[-1]:
            return None
        if residual_inp_.dtype != input_.dtype or weight_.dtype != input_.dtype:
            return None
        if residual_inp_.device != input_.device or weight_.device != input_.device:
            return None
        if not is_weak_contiguous(residual_inp_) or not weight_.is_contiguous():
            return None

        if not input_.is_contiguous() or not residual_inp_.is_contiguous():
            return None
        if not self._should_fused_rmsnorm_custom_ar(input_):
            return None
        rows, hidden = input_.shape
        if hidden % 8 != 0:
            return None
        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        graph_registered_input_enabled = getattr(
            self, "_graph_registered_input_enabled", False
        )
        use_row_single_kernel = (
            (
                self.world_size == 2
                and _fused_rmsnorm_tp2_row_one_shape(int(rows), int(hidden))
                and (
                    not _fused_rmsnorm_tp2_single_kernel_shape(int(rows), int(hidden))
                    or (int(hidden) == 1536 and 256 <= int(rows) <= 512)
                )
            )
            or (
                self.world_size == 4
                and _fused_rmsnorm_tp4_row_one_shape(int(rows), int(hidden))
            )
            or (
                self.world_size == 8
                and _fused_rmsnorm_tp8_row_one_shape(int(rows), int(hidden))
            )
        )
        if self.world_size == 4 and int(hidden) == 8192 and int(rows) <= 16:
            use_row_single_kernel = False
        if not _fused_rmsnorm_default_shape(self.world_size, rows, hidden):
            return None

        norm_out = None
        residual_out = residual_inp_
        if use_row_single_kernel:
            rows, hidden = input_.shape
            use_h8192_sum_sidecar = hidden == 8192 and _fused_rmsnorm_default_shape(
                self.world_size, int(rows), int(hidden)
            )
            use_h8192_row_one = hidden == 8192 and (
                (
                    self.world_size == 4
                    and _fused_rmsnorm_tp4_row_one_shape(int(rows), int(hidden))
                )
                or (
                    self.world_size == 8
                    and _fused_rmsnorm_tp8_row_one_shape(int(rows), int(hidden))
                )
            )
            use_small_hidden_row_one = hidden in (512, 1024, 2048)
            use_tp8_small_row_one = self.world_size == 8 and (
                hidden == 2048
                or (hidden == 512 and (rows < 128 or rows >= 8192))
                or (hidden == 1024 and rows >= 8192)
            )
            use_tp8_row_one = (
                self.world_size == 8
                and (
                    (hidden == 4096 and rows >= 512)
                    or use_tp8_small_row_one
                    or _fused_rmsnorm_tp8_row_one_shape(int(rows), int(hidden))
                )
                and _fused_rmsnorm_tp8_row_one_shape(int(rows), int(hidden))
            )
            use_tp2_row_one = (
                self.world_size == 2
                and _fused_rmsnorm_tp2_row_one_shape(int(rows), int(hidden))
                and _fused_rmsnorm_default_shape(
                    self.world_size, int(rows), int(hidden)
                )
            )
            use_tp4_row_one_default = (
                self.world_size == 4
                and _fused_rmsnorm_tp4_row_one_shape(int(rows), int(hidden))
            )
            use_tp4_sum_sidecar = (
                (self.world_size in (4, 8) and use_h8192_sum_sidecar)
                or (
                    self.world_size == 4
                    and (hidden in (512, 1024, 2048, 4096) or use_h8192_sum_sidecar)
                )
                or use_tp4_row_one_default
                or use_tp8_row_one
                or use_tp2_row_one
            )
            sidecar_use_registered_rank_data = (
                is_graph_launch and graph_registered_input_enabled
            )
            # Align with CUDA/sgl-kernel semantics: graph decode may use
            # pre-registered activation rank data, while eager prefill stages
            # through the communicator workspace.
            if use_tp4_sum_sidecar:
                try:
                    use_row_one_kernel = (
                        use_tp4_row_one_default
                        or use_h8192_row_one
                        or use_tp8_row_one
                        or use_tp2_row_one
                    )
                    if use_row_one_kernel:
                        if norm_out is None and self.world_size == 4:
                            norm_out = torch.empty_like(input_)
                        use_tuned_module = (
                            use_tp4_row_one_default
                            and self.world_size == 4
                            and (
                                (int(rows) >= 1024 and int(hidden) == 1024)
                                or (int(rows) == 128 and int(hidden) == 4096)
                            )
                        )
                        use_small_rows_module = (
                            self.world_size == 4
                            and (
                                (int(hidden) == 1024 and int(rows) <= 128)
                                or (int(hidden) == 2048 and 32 <= int(rows) <= 128)
                                or (int(hidden) == 4096 and int(rows) <= 512)
                            )
                        ) or (
                            self.world_size == 8
                            and (
                                (int(hidden) == 512 and int(rows) <= 512)
                                or (int(hidden) == 1024 and 128 <= int(rows) <= 512)
                                or (int(hidden) == 2048 and int(rows) <= 512)
                            )
                        )
                        use_row_skip_end_barrier = (
                            self.world_size in (2, 4, 8) and int(hidden) <= 16384
                        )
                        row_token_2stage_override = 1
                        row_warp_inv_rms_override = (
                            1
                            if (
                                self.world_size == 4
                                and int(hidden) in (512, 1024, 2048, 3072)
                            )
                            else -1
                        )
                        if norm_out is None:
                            norm_out = torch.empty_like(input_)
                        launched = self._launch_fused_allreduce_rmsnorm_row(
                            input_,
                            residual_inp_,
                            residual_out,
                            norm_out,
                            weight_,
                            int(hidden),
                            use_tuned_module,
                            use_small_rows_module,
                            use_row_skip_end_barrier,
                            row_token_2stage_override,
                            row_warp_inv_rms_override,
                            float(eps),
                            sidecar_use_registered_rank_data,
                        )
                        if not launched:
                            # A registered graph-input miss must preserve the
                            # collective result. Fall back to the explicit-input
                            # row launcher, matching the normal custom-AR path.
                            launched = self._launch_fused_allreduce_rmsnorm_row(
                                input_,
                                residual_inp_,
                                residual_out,
                                norm_out,
                                weight_,
                                int(hidden),
                                use_tuned_module,
                                use_small_rows_module,
                                use_row_skip_end_barrier,
                                row_token_2stage_override,
                                row_warp_inv_rms_override,
                                float(eps),
                                False,
                            )
                        if not launched:
                            return None
                        return norm_out, residual_out
                    return None
                except Exception:
                    raise
            return None

        rows, hidden = input_.shape
        use_tp2_h4096_default = (
            self.world_size == 2 and int(hidden) == 4096 and 1 <= int(rows) <= 4096
        )
        use_tp2_h4096_short_default = (
            self.world_size == 2 and int(hidden) == 4096 and 1 <= int(rows) <= 512
        )
        use_tp2_registered_default = use_tp2_h4096_default
        use_capture_registered_default = (
            is_graph_launch and graph_registered_input_enabled
        )
        use_registered_input = use_capture_registered_default or (
            is_graph_launch
            and graph_registered_input_enabled
            and use_tp2_registered_default
        )
        use_short_rows_module = (
            (
                self.world_size == 2
                and (
                    (int(hidden) == 4096 and int(rows) < 2048)
                    or (int(hidden) == 1024 and int(rows) < 512)
                    or (
                        512 <= int(hidden) <= 8192
                        and int(hidden) % 8 == 0
                        and not _is_power_of_2(int(hidden))
                        and int(rows) <= 128
                    )
                )
                and (
                    use_tp2_h4096_short_default
                    or int(hidden) == 1024
                    or (
                        512 <= int(hidden) <= 8192
                        and int(hidden) % 8 == 0
                        and not _is_power_of_2(int(hidden))
                        and int(rows) <= 128
                    )
                )
            )
            or (
                self.world_size == 4
                and (int(hidden) == 512 or (int(hidden) == 8192 and int(rows) <= 16))
                and int(rows) < 1024
            )
            or (self.world_size == 8 and int(hidden) == 1536 and int(rows) <= 2048)
        )
        use_row_skip_end_barrier = self.world_size in (2, 4, 8) and int(hidden) <= 16384
        cache_policy = 0
        if use_registered_input and norm_out is None:
            norm_out = torch.empty_like(input_)
        row_warp_inv_rms_override = (
            1
            if (
                self.world_size == 8 and int(rows) == 256 and int(hidden) in (512, 1024)
            )
            else -1
        )
        if use_registered_input:
            rank_data_ready = False
            try:
                rank_data = self._rank_data_for_registered_input(input_)
                if rank_data is None:
                    raise RuntimeError("registered rank data is not ready")
                rank_data_ready = True
                launcher_key = (
                    int(hidden),
                    use_short_rows_module,
                    cache_policy,
                    use_row_skip_end_barrier,
                    row_warp_inv_rms_override,
                )
                launcher = self._fused_rmsnorm_registered_launchers.get(launcher_key)
                if launcher is None:
                    launcher = (
                        self._jit_ar.launch_fused_allreduce_rmsnorm_registered_func(
                            self.world_size,
                            int(hidden),
                            use_short_rows_module,
                            False,
                            cache_policy,
                            use_row_skip_end_barrier,
                            row_warp_inv_rms_override,
                        )
                    )
                    self._fused_rmsnorm_registered_launchers[launcher_key] = launcher
                launcher(
                    rank_data,
                    self.signal_ptrs_cpu,
                    residual_inp_,
                    residual_out,
                    norm_out,
                    weight_,
                    int(self.meta_ptrs[self.rank]),
                    int(self.rank),
                    int(self.world_size),
                    float(eps),
                )
                return norm_out, residual_out
            except Exception:
                if (
                    is_graph_launch
                    and graph_registered_input_enabled
                    and not rank_data_ready
                ):
                    # Registration can miss while the graph is being
                    # recaptured. Continue into the explicit-input fused
                    # launcher below instead of returning silent zeros.
                    pass
                else:
                    raise

        if norm_out is None:
            norm_out = torch.empty_like(input_)
        reset_lamport = False
        push_polling_override = 0
        lamport_push_override = 0
        launcher_key = (
            int(hidden),
            use_short_rows_module,
            cache_policy,
            use_row_skip_end_barrier,
            push_polling_override,
            lamport_push_override,
            row_warp_inv_rms_override,
        )
        launcher = self._fused_rmsnorm_unregistered_launchers.get(launcher_key)
        if launcher is None:
            launcher = self._jit_ar.launch_fused_allreduce_rmsnorm_unregistered_func(
                self.world_size,
                int(hidden),
                use_short_rows_module,
                False,
                cache_policy,
                use_row_skip_end_barrier,
                push_polling_override,
                lamport_push_override,
                row_warp_inv_rms_override,
            )
            self._fused_rmsnorm_unregistered_launchers[launcher_key] = launcher
        launcher(
            self.rank_data,
            self.signal_ptrs_cpu,
            input_,
            norm_out,
            residual_inp_,
            residual_out,
            weight_,
            int(self.meta_ptrs[self.rank]),
            int(self.buffer_ptrs[self.rank]),
            int(self.max_size),
            int(self.rank),
            int(self.world_size),
            float(eps),
            int(reset_lamport),
        )
        return norm_out, residual_out

    def _launch_fused_allreduce_rmsnorm_row(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        residual_out: torch.Tensor,
        norm_out: torch.Tensor,
        weight_: torch.Tensor,
        hidden: int,
        use_tuned_module: bool,
        use_small_rows_module: bool,
        use_row_skip_end_barrier: bool,
        token_2stage_override: int,
        row_warp_inv_rms_override: int,
        eps: float,
        use_graph_registered_rank_data: bool,
    ) -> bool:
        launcher_key = (
            int(hidden),
            bool(use_tuned_module),
            bool(use_small_rows_module),
            bool(use_row_skip_end_barrier),
            int(token_2stage_override),
            int(row_warp_inv_rms_override),
        )
        if not use_graph_registered_rank_data:
            launcher = self._fused_rmsnorm_row_unregistered_launchers.get(launcher_key)
            if launcher is None:
                launcher = (
                    self._jit_ar.launch_fused_allreduce_rmsnorm_row_unregistered_func(
                        self.world_size,
                        hidden,
                        use_tuned_module,
                        use_small_rows_module,
                        use_row_skip_end_barrier,
                        token_2stage_override,
                        row_warp_inv_rms_override,
                    )
                )
                self._fused_rmsnorm_row_unregistered_launchers[launcher_key] = launcher
            launcher(
                self.rank_data,
                self.signal_ptrs_cpu,
                input_,
                residual_inp_,
                residual_out,
                norm_out,
                weight_,
                int(self.meta_ptrs[self.rank]),
                int(self.buffer_ptrs[self.rank]),
                int(self.max_size),
                int(self.rank),
                int(self.world_size),
                int(hidden),
                float(eps),
            )
            return True

        rank_data = self._rank_data_for_registered_input(input_)
        if rank_data is None:
            return False
        launcher = self._fused_rmsnorm_row_registered_launchers.get(launcher_key)
        if launcher is None:
            launcher = self._jit_ar.launch_fused_allreduce_rmsnorm_row_registered_func(
                self.world_size,
                hidden,
                use_tuned_module,
                use_small_rows_module,
                use_row_skip_end_barrier,
                token_2stage_override,
                row_warp_inv_rms_override,
            )
            self._fused_rmsnorm_row_registered_launchers[launcher_key] = launcher
        launcher(
            rank_data,
            self.signal_ptrs_cpu,
            residual_inp_,
            residual_out,
            norm_out,
            weight_,
            int(self.meta_ptrs[self.rank]),
            int(self.rank),
            int(self.world_size),
            int(hidden),
            float(eps),
        )
        return True
