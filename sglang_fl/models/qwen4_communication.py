# SPDX-License-Identifier: Apache-2.0
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
"""Install local TP8 communication only for the Qwen3.8 model."""

import logging
import socket

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def groups():
    from sglang.srt.distributed.parallel_state import (
        get_tp_group,
        get_attn_tp_group,
        get_moe_tp_group,
    )

    unique = {}
    for group in (get_tp_group(), get_attn_tp_group(), get_moe_tp_group()):
        unique[id(group)] = group
    return tuple(unique.values())


def prepare():
    from sglang_fl.distributed.device_communicators.musa_custom_ar import (
        MusaCustomAllreduce,
    )

    for group in groups():
        if (
            group.world_size != 8
            or group.device.type != "musa"
            or not group.use_custom_allreduce
        ):
            continue
        if isinstance(group.ca_comm, MusaCustomAllreduce):
            continue
        devices = [None] * group.world_size
        properties = torch.musa.get_device_properties(group.device)
        dist.all_gather_object(
            devices,
            (socket.gethostname(), group.device.index, properties.name),
            group=group.cpu_group,
        )
        if len({entry[0] for entry in devices}) != 1 or any(
            entry[2] != "MTT S5000" for entry in devices
        ):
            continue
        can_access = all(
            peer == group.device.index
            or torch.musa.can_device_access_peer(group.device, peer)
            for _, peer, _ in devices
        )
        peer_access = [None] * group.world_size
        dist.all_gather_object(peer_access, can_access, group=group.cpu_group)
        if not all(peer_access):
            continue
        framework_communicator = getattr(group, "fl_communicator", None)
        if framework_communicator is not None:
            flagcx = framework_communicator._flagcx_comm
            if flagcx is not None and not flagcx.disabled:
                flagcx._use_inplace_all_reduce = True
        if group.ca_comm is not None:
            group.ca_comm.close()
        group.ca_comm = MusaCustomAllreduce(group.cpu_group, group.device)
        group._fl_qwen38_custom_ar = group.ca_comm
        logger.info(
            "Prepared Qwen3.8 custom AllReduce for local S5000 TP8: "
            "BF16 batch-64 decode graphs, hidden widths 2560 and 10240; "
            "validated FlagCX messages use in-place output"
        )


def _destroy(original, self):
    communicator = getattr(self, "_fl_qwen38_custom_ar", None)
    if communicator is not None:
        communicator.close()
        self._fl_qwen38_custom_ar = None
        self.ca_comm = None
    return original(self)


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.distributed.parallel_state.GroupCoordinator.destroy",
            _destroy,
            HookType.AROUND,
        )
    ]
