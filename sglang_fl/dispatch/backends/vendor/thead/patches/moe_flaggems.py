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

"""Route sglang's hard-wired sgl_kernel MoE helpers to FlagGems (Triton) on PPU.

On ``is_cuda()`` platforms sglang calls ``sgl_kernel.moe_align_block_size`` (via
its wrapper in ``triton_utils/moe_align_block_size.py``) and
``sgl_kernel.moe_sum_reduce`` directly, bypassing the plugin dispatcher. This
patch swaps them for ``flag_gems.moe_align_block_size`` / ``flag_gems.moe_sum``.

Opt-in: ``SGLANG_FL_THEAD_MOE_FLAGGEMS=1``. Not used with expert parallelism
(ep_size > 1), where topk_ids contain -1 for filtered experts.

Scheduler processes import ``fused_moe`` before ``load_plugins()`` runs, so the
already-bound references in loaded ``sglang.srt`` modules are rebound as well.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_ALIGN_MOD = "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
_FUSED_MOE_MOD = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
# Modules whose functions do lazy `from ... import moe_align_block_size`.
_EXTRA_MODS = (
    "sglang.srt.layers.moe.moe_runner.triton_utils",
    "sglang.srt.layers.moe.fused_moe_triton",
    "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe",
)

_use_ep = None


def _ep_enabled() -> bool:
    # Checked lazily: server args are not configured yet when the plugin loads.
    global _use_ep
    if _use_ep is None:
        try:
            from sglang.srt.server_args import get_global_server_args

            _use_ep = get_global_server_args().ep_size > 1
        except Exception:
            return False  # not configured yet; decide on a later call
    return _use_ep


def _rebind(old, new) -> list:
    """Replace attributes equal to ``old`` in loaded sglang.srt modules."""
    hits = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("sglang.srt."):
            continue
        for attr, val in list(vars(mod).items()):
            if val is old:
                setattr(mod, attr, new)
                hits.append(f"{name}.{attr}")
    return hits


def patch_moe_flaggems() -> None:
    if os.environ.get("SGLANG_FL_THEAD_MOE_FLAGGEMS", "0") != "1":
        return

    import importlib

    import flag_gems
    import torch

    # FlagGems tries a TLE kernel first on every call; PPU lacks
    # tle.distributed_barrier, so each call fails (~31 ms) before falling back
    # to the plain Triton path (~0.1 ms). HAS_TLE is read at call time.
    fg_align = sys.modules.get("flag_gems.fused.moe_align_block_size")
    if fg_align is not None and getattr(fg_align, "HAS_TLE", False):
        fg_align.HAS_TLE = False

    align_mod = importlib.import_module(_ALIGN_MOD)
    fused_moe = importlib.import_module(_FUSED_MOE_MOD)
    for name in _EXTRA_MODS:
        try:
            importlib.import_module(name)
        except ImportError:
            pass

    orig_align = align_mod.moe_align_block_size

    def moe_align_block_size(topk_ids, block_size, num_experts):
        if _ep_enabled():
            return orig_align(topk_ids, block_size, num_experts)
        return flag_gems.moe_align_block_size(topk_ids, block_size, num_experts)

    orig_sum = getattr(fused_moe, "moe_sum_reduce", None)

    def moe_sum_reduce(input_tensor, output_tensor, routed_scaling_factor=1.0):
        if routed_scaling_factor in (None, 0, 1.0):
            flag_gems.moe_sum(input_tensor, output_tensor)
            return
        # Not in-place: FlagGems autotunes scalar mul_ on the real tensor without
        # restoring it, so a cold-cache mul_ applies the scale many times.
        tmp = torch.empty_like(output_tensor)
        flag_gems.moe_sum(input_tensor, tmp)
        torch.mul(tmp, routed_scaling_factor, out=output_tensor)

    hits = _rebind(orig_align, moe_align_block_size)
    if orig_sum is not None:
        # Only sglang.srt modules are rebound; sgl_kernel itself is untouched.
        hits += _rebind(orig_sum, moe_sum_reduce)
    logger.info("thead: MoE helpers -> FlagGems: %s", ", ".join(hits) or "none")
