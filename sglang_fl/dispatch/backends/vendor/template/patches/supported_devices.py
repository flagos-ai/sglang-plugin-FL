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

"""Extend sglang's SUPPORTED_DEVICES with this vendor's device_type."""

import logging

logger = logging.getLogger(__name__)


def patch_supported_devices():
    from sglang.srt.configs import device_config as dc

    # Replace "template" with your torch device_type (e.g. "musa", "npu", "gcu").
    if "template" not in dc.SUPPORTED_DEVICES:
        dc.SUPPORTED_DEVICES = [*dc.SUPPORTED_DEVICES, "template"]
        logger.info("patched SUPPORTED_DEVICES += [template]")
