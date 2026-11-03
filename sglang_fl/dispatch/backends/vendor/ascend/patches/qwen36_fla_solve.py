"""Use fixed-length FLA kernels for a guarded Qwen3.6-27B prefill."""

from contextvars import ContextVar
import functools
import inspect
import logging
import os

import torch

from .fia_tnd_gqa import _cpu_ints

logger = logging.getLogger(__name__)
_active_cu = ContextVar("sglang_fl_fla_solve_cu", default=None)
_marker = "_sglang_fl_fla_single_solve"
_logged_stages = set()


def _patch_other_stages(chunk):
    """Skip the variable-length path for compatible stages, but retain W/U."""
    stages = (
        ("chunk_local_cumsum", "g", (1, 16384, 24)),
        ("chunk_scaled_dot_kkt_fwd", "k", (1, 16384, 8, 128)),
        ("chunk_gated_delta_rule_fwd_h", "k", (1, 16384, 8, 128)),
        ("chunk_fwd_o", "q", (1, 16384, 8, 128)),
    )
    originals = [
        (name, getattr(chunk, name), arg, shape) for name, arg, shape in stages
    ]
    if any(
        "cu_seqlens" not in inspect.signature(fn).parameters
        for _, fn, _, _ in originals
    ):
        logger.warning("FLA fixed-length stages skipped: incompatible installed API")
        return False

    for name, original, tensor_arg, shape in originals:

        @functools.wraps(original)
        def wrapped(
            *args,
            _original=original,
            _arg=tensor_arg,
            _shape=shape,
            _name=name,
            **kwargs,
        ):
            expected_cu = _active_cu.get()
            cu = kwargs.get("cu_seqlens")
            tensor = kwargs.get(_arg, args[0] if args else None)
            if (
                expected_cu is not None
                and isinstance(cu, torch.Tensor)
                and isinstance(tensor, torch.Tensor)
                and cu.data_ptr() == expected_cu.data_ptr()
                and tuple(tensor.shape) == _shape
            ):
                if _name not in _logged_stages:
                    logger.info("FLA_FIXED_LENGTH_HIT: %s", _name)
                    _logged_stages.add(_name)
                kwargs["cu_seqlens"] = None
            return _original(*args, **kwargs)

        setattr(chunk, name, wrapped)
    return True


def _eligible(self, layer, forward_batch, mixed_qkv):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return (
        forward_batch.forward_mode == ForwardMode.EXTEND
        and forward_batch.spec_algorithm.is_none()
        and forward_batch.batch_size == 1
        and _cpu_ints(forward_batch.extend_seq_lens_cpu) == [16384]
        and _cpu_ints(forward_batch.seq_lens_cpu) == [16384]
        and _cpu_ints(getattr(forward_batch, "extend_prefix_lens_cpu", None))
        in (None, [0])
        and not self.graph_mode
        and tuple(mixed_qkv.shape) == (16384, 5120)
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.device.type == "npu"
        and (layer.num_q_heads, layer.num_v_heads) == (8, 24)
        and (layer.head_q_dim, layer.head_v_dim) == (128, 128)
        and not torch.npu.is_current_stream_capturing()
    )


def patch_qwen36_fla_solve():
    if os.getenv("SGLANG_FL_FLA_SINGLE_SOLVE", "0") != "1":
        return False
    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
            AscendGDNAttnBackend,
        )
        from sgl_kernel_npu.fla import chunk
    except ImportError as exc:
        logger.warning("FLA single solve unavailable: %s", exc)
        return False

    original_forward = AscendGDNAttnBackend.forward_extend
    original_solve = chunk.solve_tril
    if getattr(original_forward, _marker, False):
        return False
    if tuple(inspect.signature(original_forward).parameters) != (
        "self",
        "layer",
        "forward_batch",
        "mixed_qkv",
        "a",
        "b",
        "kwargs",
    ) or tuple(inspect.signature(original_solve).parameters) != (
        "A",
        "cu_seqlens",
        "output_dtype",
    ):
        logger.warning("FLA single solve skipped: incompatible installed API")
        return False

    @functools.wraps(original_forward)
    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        if not _eligible(self, layer, forward_batch, mixed_qkv):
            return original_forward(
                self, layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        token = _active_cu.set(self.forward_metadata.query_start_loc)
        try:
            return original_forward(
                self, layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        finally:
            _active_cu.reset(token)

    @functools.wraps(original_solve)
    def solve_tril(A, cu_seqlens=None, output_dtype=torch.float):
        expected_cu = _active_cu.get()
        if (
            expected_cu is not None
            and isinstance(cu_seqlens, torch.Tensor)
            and cu_seqlens.data_ptr() == expected_cu.data_ptr()
            and tuple(cu_seqlens.shape) == (2,)
            and cu_seqlens.dtype == torch.int32
            and tuple(A.shape) == (1, 16384, 24, 64)
            and A.dtype == torch.float32
            and A.device.type == "npu"
            and output_dtype == torch.bfloat16
        ):
            return original_solve(A, None, output_dtype)
        return original_solve(A, cu_seqlens, output_dtype)

    setattr(forward_extend, _marker, True)
    AscendGDNAttnBackend.forward_extend = forward_extend
    chunk.solve_tril = solve_tril
    if os.getenv("SGLANG_FL_FLA_HYBRID_CHAIN", "0") == "1":
        _patch_other_stages(chunk)
    logger.info("Installed opt-in single-segment FLA triangular solve")
    return True
