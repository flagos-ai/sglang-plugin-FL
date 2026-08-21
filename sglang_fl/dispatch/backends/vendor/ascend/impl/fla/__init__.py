# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Ascend NPU FLA kernels, mirroring `sgl_kernel_npu.fla` file-for-file.
#
# Long-term vendored (Ascend-only, no upstream replacement expected):
#   - fused_gdn_gating.py
#   - fused_sigmoid_gating_recurrent.py
#   - layernorm_gated.py
#
# Adapter (routes through plugin dispatch fallback to flag_gems / reference):
#   - chunk.py
#
# Shared helper:
#   - utils.py (only `input_guard`)
#
# Upstream: https://github.com/sgl-project/sgl-kernel-npu/tree/main/python/sgl_kernel_npu/sgl_kernel_npu/fla
