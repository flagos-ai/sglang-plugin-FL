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

"""T-Head PPU monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
Add one ``patch_xxx`` call per concern; put the implementation under ``patches/``.
"""

import logging

from .patches.rs_norm_ag import patch_rs_norm_ag

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_thead_patches():
    """Apply all T-Head PPU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_rs_norm_ag()


apply_thead_patches()
