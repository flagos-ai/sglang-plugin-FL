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

"""Vendor patches that must run before sglang_fl imports sglang itself.

Split out from ``patch.py`` because load_plugin() imports sglang modules from
its own layers — the dispatch AROUND hook builds op classes — so a fix whose
whole point is to make those imports survivable cannot wait for the late
vendor slot. Everything here is a prerequisite of *using* sglang on this
vendor; anything that only adjusts behaviour belongs in ``patch.py``.
"""

from .patches.flashinfer_stub import patch_flashinfer_stub

patch_flashinfer_stub()
