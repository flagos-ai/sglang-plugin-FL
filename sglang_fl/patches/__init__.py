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

"""Vendor-independent monkey-patches on sglang internals — entrypoint.

Called by ``sglang_fl.load_plugin()`` (see ``_apply_common_patches``) before
the vendor patches, so ``vendor/<name>/patch.py`` can still override them.
Use this for fixes that apply to every vendor, e.g. backports of upstream
SGLang bug fixes missing from the pinned base version. Add one ``patch_xxx``
call per concern; put the implementation in a sibling module.
"""

from .mamba_pool import patch_mamba_pool_copy_from

_patches_applied = False


def apply_common_patches() -> None:
    """Apply all vendor-independent patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_mamba_pool_copy_from()
