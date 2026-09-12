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

"""Vendor monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
Add one ``patch_xxx`` call per concern; put the implementation under ``patches/``.
"""

import logging

from .patches.attention_backend_choice import patch_attention_backend_choice
from .patches.causal_conv1d import patch_causal_conv1d
from .patches.clamp_position import patch_clamp_position
from .patches.jit_kernels import patch_jit_kernel_predicates
from .patches.pp_send_first import patch_pp_send_recv_and_preprocess_output_tensors
from .patches.suppress_pynccl import patch_suppress_pynccl

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_kunlunxin_patches():
    """Apply all kunlunxin-specific patches.

    Each patch is isolated: one that no longer matches sglang's API must not
    take the rest down with it. That is not hypothetical — a rebind left over
    from an older sglang raised AttributeError here and silently disabled every
    patch behind it, including the ones this backend needs to run at all.
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    for name, patch in (
        ("clamp_position", patch_clamp_position),
        ("causal_conv1d", patch_causal_conv1d),
        ("jit_kernels", patch_jit_kernel_predicates),
        ("suppress_pynccl", patch_suppress_pynccl),
        ("pp_send_recv", patch_pp_send_recv_and_preprocess_output_tensors),
        ("attention_backend_choice", patch_attention_backend_choice),
    ):
        try:
            patch()
        except Exception as e:
            logger.warning("kunlunxin patch %s failed (continuing): %r", name, e)


apply_kunlunxin_patches()
