# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from .patches.clamp_position import patch_clamp_position
from .patches.legacy_gpu_gate import patch_legacy_gpu_gate
from .patches.triton_pdl import patch_triton_pdl_symbols

patch_triton_pdl_symbols()
patch_clamp_position()
patch_legacy_gpu_gate()
