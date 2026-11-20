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

"""T-Head PPU patches that must run before SGLang imports its layers.

Auto-imported by ``sglang_fl.activate_platform()`` (see
``_apply_vendor_early_patches``). Keep imports here free of ``sglang.srt``.
"""

from .import_guard import install_import_guards

install_import_guards()
