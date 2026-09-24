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

"""Unit coverage for the minimal SGLang 0.5.18 Ascend PP patch."""

import sys
from types import ModuleType, SimpleNamespace


def _install_fake_scheduler(monkeypatch):
    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    managers = ModuleType("sglang.srt.managers")
    pp_module = ModuleType("sglang.srt.managers.scheduler_pp_mixin")

    class SchedulerPPMixin:
        def _pp_send_recv_and_preprocess_output_tensors(
            self,
            next_first_rank_mb_id,
            next_mb_id,
            mbs,
            mb_metadata,
            last_rank_comm_queue,
            pp_outputs,
        ):
            return (not pp_module.is_xpu()) or self.ps.pp_rank % 2 == 0

        def _pp_launch_batch(
            self,
            mb_id,
            cur_batch,
            pp_proxy_tensors,
            mb_metadata,
            last_rank_comm_queue,
        ):
            return "result", "event"

    pp_module.SchedulerPPMixin = SchedulerPPMixin
    pp_module.is_xpu = lambda: False
    managers.scheduler_pp_mixin = pp_module
    srt.managers = managers
    sglang.srt = srt

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers)
    monkeypatch.setitem(sys.modules, pp_module.__name__, pp_module)
    return pp_module, SchedulerPPMixin


def test_pp_exchange_uses_rank_parity_and_preserves_upstream_body(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.scheduler_pp import (
        patch_pp_send_recv_order,
    )

    pp_module, mixin = _install_fake_scheduler(monkeypatch)
    patch_pp_send_recv_order()

    syncs = []
    instance = mixin()
    instance.device_module = SimpleNamespace(synchronize=lambda: syncs.append(True))
    call_args = (0, 0, [], [], [], None)

    instance.ps = SimpleNamespace(pp_rank=0)
    assert instance._pp_send_recv_and_preprocess_output_tensors(*call_args) is True
    instance.ps = SimpleNamespace(pp_rank=1)
    assert instance._pp_send_recv_and_preprocess_output_tensors(*call_args) is False
    assert syncs == [True, True]
    assert pp_module.is_xpu() is True

    patched = mixin._pp_send_recv_and_preprocess_output_tensors
    patch_pp_send_recv_order()
    assert mixin._pp_send_recv_and_preprocess_output_tensors is patched


def test_pp_launch_returns_upstream_value_after_forward_sync(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.scheduler_pp import (
        patch_pp_launch_batch_sync,
    )

    _, mixin = _install_fake_scheduler(monkeypatch)
    patch_pp_launch_batch_sync()

    syncs = []
    instance = mixin()
    instance.forward_stream = SimpleNamespace(synchronize=lambda: syncs.append(True))
    result = instance._pp_launch_batch(0, object(), object(), [], [])

    assert result == ("result", "event")
    assert syncs == [True]


def test_pp_patch_rejects_unknown_scheduler_signature(monkeypatch) -> None:
    import pytest

    from sglang_fl.dispatch.backends.vendor.ascend.patches.scheduler_pp import (
        patch_pp_send_recv_order,
    )

    pp_module, mixin = _install_fake_scheduler(monkeypatch)

    def incompatible(self, renamed_parameter):
        return None

    mixin._pp_send_recv_and_preprocess_output_tensors = incompatible
    with pytest.raises(RuntimeError, match="Unsupported SGLang scheduler interface"):
        patch_pp_send_recv_order()
    assert pp_module.is_xpu() is False
