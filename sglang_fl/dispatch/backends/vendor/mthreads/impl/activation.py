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

# MUSA activation adapters backed by the vendor SGLang JIT kernels.


from __future__ import annotations

import torch


def silu_and_mul_musa(obj, x: torch.Tensor) -> torch.Tensor:
    """Run the native MUSA JIT without re-entering the OOT layer wrapper."""
    if x.device.type != "musa":
        raise NotImplementedError("MUSA SiLU+Mul requires a MUSA tensor")
    if x.ndim < 1 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("SiLU+Mul requires a positive even last dimension")
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_contiguous():
        raise NotImplementedError("MUSA JIT SiLU+Mul requires contiguous fp16/bf16 input")
    try:
        from sglang.srt.hardware_backend.musa.jit_kernel import act_and_mul
    except ImportError as exc:
        raise NotImplementedError("This SGLang build has no native MUSA act_and_mul") from exc
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    # Empty batches need no kernel launch.
    if x.numel() == 0:
        return x.new_empty(output_shape)
    return act_and_mul(x.view(-1, x.shape[-1]), activation="silu").view(output_shape)
