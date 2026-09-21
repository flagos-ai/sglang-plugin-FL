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

"""SGLang 0.5.18 Ascend MoE dispatch adapter."""

from __future__ import annotations


def fused_moe_ascend(obj, layer, dispatch_output):
    """Delegate to SGLang's NPU runner without re-dispatching tokens.

    SGLang 0.5.18 performs Ascend routing before the fused-op method and passes
    an ``AscendTPDispatchOutput`` here. The pre-0.5.18 plugin implementation
    expected a ``StandardDispatchOutput`` and routed the tokens a second time;
    besides doing duplicate work, it accessed a removed ``topk_output`` field.
    The native 0.5.18 method owns that runner/dispatcher contract, so the
    plugin delegates to it rather than duplicating version-sensitive details.
    """

    forward_npu = getattr(obj, "forward_npu", None)
    if forward_npu is None:
        raise RuntimeError(
            "SGLang 0.5.18 UnquantizedFusedMoEMethod.forward_npu is required by "
            "the Ascend fused MoE adapter"
        )
    return forward_npu(layer, dispatch_output)
