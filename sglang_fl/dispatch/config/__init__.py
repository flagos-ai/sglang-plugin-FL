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

# Hardware-specific operator configuration loader.
#
# This module provides automatic loading of operator configurations based on
# the detected hardware platform. See utils.py for implementation details.

from sglang_fl.dispatch.config.utils import (
    get_config_path,
    get_effective_config,
    get_flagos_blacklist,
    get_oot_blacklist,
    get_per_op_order,
    get_platform_name,
    load_platform_config,
)

__all__ = [
    "get_platform_name",
    "get_config_path",
    "load_platform_config",
    "get_per_op_order",
    "get_flagos_blacklist",
    "get_oot_blacklist",
    "get_effective_config",
]
