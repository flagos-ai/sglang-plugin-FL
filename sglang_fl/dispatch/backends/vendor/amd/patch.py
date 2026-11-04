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

"""AMD ROCm patches for sglang's Triton fused-MoE runner.

On ROCm, sglang's fused MoE calls sgl_kernel for moe_align_block_size and the
gated activations. sgl_kernel's ROCm build only targets the archs below, so on
other AMD GPUs (e.g. Radeon gfx1201) these calls are routed to FlagGems' Triton
kernels instead. The arch is checked on first call, when the device is known.
"""

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_SGL_KERNEL_ROCM_ARCHS = ("gfx942", "gfx950")
_use_flaggems_cache = {}


def _use_flaggems(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _use_flaggems_cache:
        arch = torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]
        _use_flaggems_cache[index] = arch not in _SGL_KERNEL_ROCM_ARCHS
        if _use_flaggems_cache[index]:
            logger.info(
                "fused MoE on %s: using FlagGems kernels instead of sgl_kernel", arch
            )
    return _use_flaggems_cache[index]


def _wrap_moe_align_block_size(original):
    def moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer=None,
        pad_sorted_token_ids=False,
    ):
        if not _use_flaggems(topk_ids.device):
            return original(
                topk_ids,
                num_experts,
                block_size,
                sorted_token_ids,
                experts_ids,
                num_tokens_post_pad,
                cumsum_buffer,
                pad_sorted_token_ids,
            )
        from flag_gems.fused.moe_align_block_size import moe_align_block_size_triton

        # FlagGems always fills unused sorted_token_ids slots with topk_ids.numel().
        moe_align_block_size_triton(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
        )

    return moe_align_block_size


def _wrap_silu_and_mul(original):
    def silu_and_mul(input, out, **kwargs):
        if kwargs or not _use_flaggems(input.device):
            return original(input, out, **kwargs)
        from flag_gems.fused.silu_and_mul import silu_and_mul_out

        d = input.shape[-1] // 2
        return silu_and_mul_out(input[..., :d], input[..., d:], out)

    return silu_and_mul


def _wrap_gelu_and_mul(original):
    def gelu_and_mul(input, out, **kwargs):
        if kwargs or not _use_flaggems(input.device):
            return original(input, out, **kwargs)
        from flag_gems.fused.gelu_and_mul import gelu_and_mul as gems_gelu_and_mul

        d = input.shape[-1] // 2
        out.copy_(gems_gelu_and_mul(input[..., :d], input[..., d:], "none"))
        return out

    return gelu_and_mul


_PATCHES = (
    ("moe_align_block_size", "sgl_moe_align_block_size", _wrap_moe_align_block_size),
    ("fused_moe", "silu_and_mul", _wrap_silu_and_mul),
    ("fused_moe", "gelu_and_mul", _wrap_gelu_and_mul),
)


def patch() -> None:
    if torch.version.hip is None:
        return
    # import_module, not "import a.b.c as m": the triton_utils package re-exports
    # functions with the same names as these submodules.
    pkg = "sglang.srt.layers.moe.moe_runner.triton_utils"
    for module_name, attr, wrap in _PATCHES:
        try:
            module = importlib.import_module(f"{pkg}.{module_name}")
            original = getattr(module, attr)
        except (ImportError, AttributeError) as e:
            logger.warning(
                "AMD fused MoE patch: %s.%s not patched (%s)", module_name, attr, e
            )
            continue
        if getattr(original, "_flagos_amd_patched", False) is True:
            continue
        wrapped = wrap(original)
        wrapped._flagos_amd_patched = True
        setattr(module, attr, wrapped)
    logger.info("AMD fused MoE patch applied")


patch()
