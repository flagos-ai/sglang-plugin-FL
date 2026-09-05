# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_patches_applied = False

_MUSA_FP32_TP_ALLREDUCE = os.environ.get(
    "SGLANG_MUSA_FP32_TP_ALLREDUCE", "0"
).lower() in ("1", "true")


def _patch_pp_send_recv_order() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    except Exception as e:
        logger.warning("MUSA PP send/recv order patch skipped: %s", e)
        return

    import torch

    orig_fn = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors

    @wraps(orig_fn)
    def pp_send_recv_and_preprocess_output_tensors_with_musa_order(
        self,
        next_first_rank_mb_id,
        next_mb_id,
        mbs,
        mb_metadata,
        last_rank_comm_queue,
        pp_outputs,
    ):
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            if mbs[next_mb_id] is None or mbs[next_mb_id].forward_mode.is_prebuilt():
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if (self.pp_rank % 2) == 0:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = (
        pp_send_recv_and_preprocess_output_tensors_with_musa_order
    )
    logger.info("MUSA PP send/recv ordering patch applied")


def _patch_pp_launch_batch_add_sync() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    except Exception as e:
        logger.warning("MUSA PP launch sync patch skipped: %s", e)
        return

    orig_fn = SchedulerPPMixin._pp_launch_batch

    @wraps(orig_fn)
    def pp_launch_batch_with_forward_stream_sync(self, *args, **kwargs):
        result, event = orig_fn(self, *args, **kwargs)
        self.forward_stream.synchronize()
        return result, event

    SchedulerPPMixin._pp_launch_batch = pp_launch_batch_with_forward_stream_sync
    logger.info("MUSA PP launch forward_stream sync patch applied")


def _patch_multimodal_mask() -> None:
    try:
        from sglang.srt.managers import mm_utils
    except Exception as e:
        logger.warning("MUSA multimodal mask patch skipped: %s", e)
        return

    import torch

    def _get_multimodal_mask_loop(
        input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token in placeholder_tensor:
            mask |= input_ids == token
        return mask.unsqueeze(-1)

    mm_utils._get_multimodal_mask = _get_multimodal_mask_loop
    logger.info("MUSA multimodal mask patch applied")


def _patch_communication_op_fp32_all_reduce() -> None:
    """Patch communication_op.py TP all-reduce functions to use FP32 accumulation.

    Wraps tensor_model_parallel_all_reduce, attention_tensor_model_parallel_all_reduce,
    and moe_tensor_model_parallel_all_reduce to convert BF16 inputs to FP32 before
    all-reduce and convert back to BF16 after.

    Also patches imported references in all consuming modules.
    """
    try:
        import sglang.srt.distributed as dist_pkg
        import sglang.srt.distributed.communication_op as comm_op
        import sglang.srt.layers.communicator as communicator
    except Exception as e:
        logger.warning("MUSA FP32 communication_op patch skipped: %s", e)
        return

    import torch

    def _tp_all_reduce_fp32(group, input_: torch.Tensor) -> torch.Tensor:
        if input_.dtype == torch.bfloat16:
            reduced = group.all_reduce(input_.float())
            return reduced.to(dtype=input_.dtype)
        return group.all_reduce(input_)

    def tensor_model_parallel_all_reduce_fp32(input_: torch.Tensor) -> torch.Tensor:
        from sglang.srt.distributed.parallel_state import get_tp_group

        return _tp_all_reduce_fp32(get_tp_group(), input_)

    def attention_tensor_model_parallel_all_reduce_fp32(
        input_: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        return _tp_all_reduce_fp32(get_attn_tp_group(), input_)

    def moe_tensor_model_parallel_all_reduce_fp32(
        input_: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.distributed.parallel_state import get_moe_tp_group

        return _tp_all_reduce_fp32(get_moe_tp_group(), input_)

    # Patch source module, the re-exported references in the package __init__, and
    # the names already bound by eager importers.
    for mod in (comm_op, dist_pkg, communicator):
        mod.tensor_model_parallel_all_reduce = tensor_model_parallel_all_reduce_fp32
        mod.attention_tensor_model_parallel_all_reduce = (
            attention_tensor_model_parallel_all_reduce_fp32
        )
        mod.moe_tensor_model_parallel_all_reduce = (
            moe_tensor_model_parallel_all_reduce_fp32
        )

    _modules_to_patch = [
        "sglang.srt.lora.layers",
        "sglang.srt.lora.triton_ops.fused_moe_lora_kernel",
    ]
    for mod_name in _modules_to_patch:
        try:
            mod = __import__(mod_name, fromlist=[""])
        except ImportError:
            continue
        if hasattr(mod, "tensor_model_parallel_all_reduce"):
            mod.tensor_model_parallel_all_reduce = tensor_model_parallel_all_reduce_fp32

    logger.info("MUSA FP32 communication_op all-reduce patch applied")


def _patch_parallel_state_fp32_reduce_scatter() -> None:
    """Patch GroupCoordinator.reduce_scatter_tensor to use FP32 accumulation.

    Converts BF16 inputs to FP32 before reduce-scatter and converts back to BF16 after.
    """
    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator
    except Exception as e:
        logger.warning("MUSA FP32 parallel_state patch skipped: %s", e)
        return

    import torch

    orig_reduce_scatter_tensor = GroupCoordinator.reduce_scatter_tensor

    @wraps(orig_reduce_scatter_tensor)
    def reduce_scatter_tensor_fp32(self, output: torch.Tensor, input: torch.Tensor):
        if input.dtype == torch.bfloat16:
            fp32_output = torch.empty_like(output, dtype=torch.float32)
            self._reduce_scatter_tensor(fp32_output, input.float())
            output.copy_(fp32_output.to(dtype=output.dtype))
            return
        orig_reduce_scatter_tensor(self, output, input)

    GroupCoordinator.reduce_scatter_tensor = reduce_scatter_tensor_fp32
    logger.info("MUSA FP32 parallel_state reduce_scatter patch applied")


def _patch_fp32_tp_all_reduce() -> None:
    """Apply all FP32 TP all-reduce patches when SGLANG_MUSA_FP32_TP_ALLREDUCE=1."""
    if not _MUSA_FP32_TP_ALLREDUCE:
        logger.info(
            "MUSA FP32 TP all-reduce: disabled "
            "(set SGLANG_MUSA_FP32_TP_ALLREDUCE=1 to enable)"
        )
        return

    _patch_communication_op_fp32_all_reduce()
    _patch_parallel_state_fp32_reduce_scatter()
    logger.info("MUSA FP32 TP all-reduce patches applied")


def _patch_device_support() -> None:
    """Teach sglang.srt.utils.common to recognise torch_musa (no torchada)."""
    try:
        from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
            patch_device_support,
        )

        patch_device_support()
    except Exception as e:
        logger.warning("MUSA device-support patch skipped: %s", e)


def _patch_flash_attn_interface() -> None:
    """Let MUSA-gated flash_attn_interface import sites survive when the
    package is absent (vision.py / musa fa3 backend)."""
    try:
        from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
            patch_flash_attn_interface,
        )

        patch_flash_attn_interface()
    except Exception as e:
        logger.warning("MUSA flash_attn_interface guard skipped: %s", e)


def _patch_attention_backend_default() -> None:
    """Default mthreads attention backend to torch_native, not fa3.

    PlatformFL's shared ``_ATTN_BACKEND_MAP`` routes mthreads to ``fa3``, but
    the flagos mthreads runtime ships no Moore Threads flash-attention-v3
    interface wheel, so the fa3 backend calls the stub's
    ``flash_attn_varlen_func`` and raises NotImplementedError on the first
    forward.  Cambricon (also PrivateUse1, no flash-attn wheel) is absent from
    the map and falls through to torch_native (PyTorch SDPA) — mthreads should
    behave the same.  The shared map stays untouched so an explicit
    ``--attention-backend fa3`` (a deliberate user choice) still resolves to
    fa3; only the value filled when the user did not pick a backend is
    rewritten.
    """
    try:
        from sglang_fl.platform import PlatformFL

        orig = PlatformFL.get_default_attention_backend

        @wraps(orig)
        def get_default_attention_backend_musa(self):
            backend = orig(self)
            if getattr(self, "_vendor_name", None) == "mthreads" and backend == "fa3":
                return "torch_native"
            return backend

        PlatformFL.get_default_attention_backend = get_default_attention_backend_musa
    except Exception as e:
        logger.warning("MUSA attention-backend default patch skipped: %s", e)


def apply_musa_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return

    _patch_flash_attn_interface()
    _patch_attention_backend_default()
    _patch_device_support()
    _patch_pp_send_recv_order()
    _patch_pp_launch_batch_add_sync()
    _patch_multimodal_mask()
    _patch_fp32_tp_all_reduce()
    _patches_applied = True
    logger.info("All MUSA PP patches applied successfully")


apply_musa_patches()
