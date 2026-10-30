"""Opt-in, shape-verified Ascend prefill tuning with untouched decode paths."""

from __future__ import annotations

import functools
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _patch_ephemeral_nz() -> bool:
    if os.getenv("SGLANG_FL_PREFILL_NZ_GATE_UP", "0") != "1":
        return False
    try:
        import torch_npu
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    except ImportError:
        logger.warning("Skipping prefill NZ: required Ascend linear API is unavailable")
        return False
    if envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
        return False

    original = UnquantizedLinearMethod.apply
    marker = "_sglang_fl_prefill_nz_gate_up"
    if getattr(original, marker, False):
        return False
    try:
        compatible = tuple(inspect.signature(original).parameters) == (
            "self",
            "layer",
            "x",
            "bias",
        )
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        logger.warning("Skipping prefill NZ: unsupported linear signature")
        return False

    @functools.wraps(original)
    def wrapped(self, layer, x, bias=None):
        weight = getattr(layer, "weight", None)
        eligible = (
            type(self) is UnquantizedLinearMethod
            and x.ndim == 2
            and tuple(x.shape) == (16384, 5120)
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and bias is None
            and weight is not None
            and tuple(weight.shape) == (17408, 5120)
            and weight.device == x.device
            and weight.dtype == x.dtype
            and weight.is_contiguous()
            and not weight.requires_grad
        )
        if not eligible or torch_npu.get_npu_format(weight) != 2:
            return original(self, layer, x, bias)
        # CANN's ND-to-NZ Identity conversion is not graph-capturable. The
        # measured decode graphs never select this prefill shape; an explicit
        # prefill capture must retain the original capturable ND linear path.
        if torch.npu.is_current_stream_capturing():
            return original(self, layer, x, bias)
        # The original Parameter stays ND. No persistent duplicate and no
        # stale-weight cache: each prefill reads the current weight contents.
        packed = torch_npu.npu_format_cast(weight, 29)
        result = torch.nn.functional.linear(x, packed)
        if not getattr(self, "_sglang_fl_nz_logged", False):
            self._sglang_fl_nz_logged = True
            logger.info("PREFILL_NZ_FAST_HIT: tokens=16384 shape=17408x5120")
        return result

    setattr(wrapped, marker, True)
    UnquantizedLinearMethod.apply = wrapped
    return True


def _patch_gating() -> bool:
    if os.getenv("SGLANG_FL_GDN_GATING_BLOCK24", "0") != "1":
        return False
    try:
        from triton.runtime import driver
        from sgl_kernel_npu.fla import fused_gdn_gating as module
        from sglang.srt.hardware_backend.npu.attention import ascend_gdn_backend
    except ImportError:
        logger.warning("Skipping gating24: required Ascend gating API is unavailable")
        return False

    original = ascend_gdn_backend.fused_gdn_gating
    marker = "_sglang_fl_prefill_gating24"
    if getattr(original, marker, False):
        return False
    kernel = getattr(module, "fused_gdn_gating_kernel", None)
    expected = (
        "g",
        "beta_output",
        "A_log",
        "a",
        "b",
        "dt_bias",
        "batch",
        "seq_len",
        "NUM_HEADS",
        "beta",
        "threshold",
        "BLK_HEADS",
    )
    try:
        compatible = tuple(inspect.signature(kernel.fn).parameters) == expected
    except (AttributeError, TypeError, ValueError):
        compatible = False
    if not compatible:
        logger.warning("Skipping gating24: unverified installed kernel ABI")
        return False

    @functools.wraps(original)
    def wrapped(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        eligible = (
            tuple(a.shape) == (16384, 24)
            and tuple(b.shape) == (16384, 24)
            and tuple(A_log.shape) == (24,)
            and tuple(dt_bias.shape) == (24,)
            and a.device.type == "npu"
            and a.device == b.device == A_log.device == dt_bias.device
            and a.dtype == b.dtype == torch.bfloat16
            and A_log.dtype == torch.float32
            and dt_bias.dtype in (torch.bfloat16, torch.float32)
            and all(x.is_contiguous() for x in (a, b, A_log, dt_bias))
        )
        if not eligible:
            return original(A_log, a, b, dt_bias, beta, threshold)
        g = torch.empty(1, 16384, 24, device=a.device, dtype=torch.float32)
        beta_out = torch.empty_like(g)
        cores = driver.active.utils.get_device_properties(torch.npu.current_device())[
            "num_vectorcore"
        ]
        kernel[(cores, 1, 1)](
            g,
            beta_out,
            A_log,
            a,
            b,
            dt_bias,
            16384,
            1,
            24,
            beta,
            threshold,
            24,
            multibuffer=True,
            num_warps=1,
        )
        if not getattr(wrapped, "_hit_logged", False):
            wrapped._hit_logged = True
            logger.info(
                "GATING24_FAST_HIT: tokens=16384 heads=24 bias_dtype=%s", dt_bias.dtype
            )
        return g, beta_out

    setattr(wrapped, marker, True)
    ascend_gdn_backend.fused_gdn_gating = wrapped
    return True


def patch_prefill_tuning() -> None:
    _patch_ephemeral_nz()
    _patch_gating()
