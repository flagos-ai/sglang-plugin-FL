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

"""Process-wide operating modes for sglang-plugin-FL."""

import os

ADAPT_MODE = "adapt"
PLATFORM_PROFILE_MODE = "platform_profile"


def get_sglang_fl_mode() -> str:
    """Return the normalized plugin mode.

    ``adapt`` preserves the existing three-layer adaptation behavior.
    ``platform_profile`` profiles the deployable target-platform baseline:
    FlagGems/FlagOS compute replacements are disabled, while plugin vendor
    dispatch and the platform runtime remain active.
    """

    raw = os.environ.get("SGLANG_FL_MODE", ADAPT_MODE).strip().lower()
    mode = raw.replace("-", "_")
    if mode not in {ADAPT_MODE, PLATFORM_PROFILE_MODE}:
        raise ValueError(
            f"Unsupported SGLANG_FL_MODE={raw!r}; expected "
            f"{ADAPT_MODE!r} or {PLATFORM_PROFILE_MODE!r}"
        )
    return mode


def is_platform_profile_mode() -> bool:
    return get_sglang_fl_mode() == PLATFORM_PROFILE_MODE
