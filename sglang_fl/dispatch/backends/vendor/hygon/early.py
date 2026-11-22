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

"""Hygon patches that must run before SGLang imports its layers.

Auto-imported by ``sglang_fl.activate_platform()`` during SGLang platform
discovery (see ``_apply_vendor_early_patches``), which the natural engine
flow reaches through ``sglang.srt.server_args`` before any
``sglang.srt.layers`` module. Keep imports here free of ``sglang.srt``.

This installs the plugin-owned replacement for the SGLang 0.5.18 HCU
source patch (see ``sglang_compat.py``); the runtime image's
``docker/hygon/sglang-v0.5.18-hcu.patch`` stays in place until this
replacement lands, a new plugin wheel is built, and the Hygon hardware
matrix passes on it.
"""

from .sglang_compat import activate as _activate_compat
from .sglang_compat import is_hcu

ACTIVATED = is_hcu() and _activate_compat()
