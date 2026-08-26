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

from sglang_fl.dispatch.backends.vendor.mthreads.patches import moe_schedule


W1_SHAPE = (256, 512, 2048)
W2_SHAPE = (256, 2048, 256)


def test_s5000_decode_shape_uses_robust_schedule(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_MOE_DECODE_SCHEDULE", raising=False)

    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return {"original": True}

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    config, (down_config, max_block_m) = wrapped(
        W1_SHAPE,
        W2_SHAPE,
        8,
        None,
        64,
        return_down_config=True,
    )

    assert config == moe_schedule._S5000_DECODE_CONFIG
    assert down_config == config
    assert down_config is not config
    assert max_block_m == 32
    assert calls == []


def test_non_target_shape_delegates(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")

    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 32) is sentinel


def test_schedule_can_be_disabled(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setenv("SGLANG_MUSA_MOE_DECODE_SCHEDULE", "off")

    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 64) is sentinel
