# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from .patches.clamp_position import patch_clamp_position
from .patches.legacy_gpu_gate import patch_legacy_gpu_gate

patch_clamp_position()
patch_legacy_gpu_gate()
