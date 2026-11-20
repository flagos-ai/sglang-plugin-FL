# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ATen and RMSNorm adapters for standalone GLM-5.2 on MUSA."""

import torch

from sglang_fl.dispatch.bridge.rms_norm import rms_norm_bridge

_aten_library = None


def register_aten_adapters(device_info, whitelist):
    """Install the exact overloads delegated outside the generic FlagGems list."""
    global _aten_library
    from flag_gems.ops.min import min as gems_min
    from flaggems_sglang import native_layer_norm

    if device_info is None or device_info.device_type != "musa":
        raise RuntimeError("The GLM-5.2 operator profile requires MUSA")
    if not {"min", "layer_norm"}.issubset(whitelist):
        raise ValueError("The GLM-5.2 profile requires min and layer_norm in its whitelist")

    def min_all(self):
        if self.numel() == 0:
            raise RuntimeError("min() requires a non-empty input for all-element reduction")
        if not self.is_contiguous():
            self = self.contiguous()
        return gems_min(self)

    library = torch.library.Library("aten", "IMPL", device_info.dispatch_key)
    library.impl("min", min_all)
    library.impl("native_layer_norm", native_layer_norm)
    _aten_library = library
    return [op for op in whitelist if op not in {"min", "layer_norm"}]


def glm52_rms_norm_bridge(self, x, residual=None, post_residual_addition=None):
    weight = self.weight
    supported = (
        type(self).__name__ == "RMSNorm"
        and getattr(self, "variance_size_override", None) is None
        and getattr(self, "override_orig_dtype", None) is None
        and not getattr(self, "fp32_residual", False)
        and not getattr(self, "cast_x_before_out_mul", False)
        and not getattr(self, "x_pad_to_multiple", 0)
        and post_residual_addition is None
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.numel() != 0
        and weight is not None
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and self.variance_epsilon > 0
    )
    if supported and residual is not None:
        supported = (
            x.ndim == 2
            and x.is_contiguous()
            and residual.is_contiguous()
            and residual.shape == x.shape
            and residual.dtype == x.dtype
            and residual.device == x.device
            and residual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        )
    if supported:
        return rms_norm_bridge(self, x, residual)
    if post_residual_addition is not None:
        return self.forward_musa(
            x, residual, post_residual_addition=post_residual_addition
        )
    return self.forward_musa(x, residual)
