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

import pytest

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


@pytest.mark.parametrize("disabled", [True, False])
def test_ineligible_runtime_does_not_inspect_shapes(monkeypatch, disabled):
    monkeypatch.setenv(moe_schedule._ENV_NAME, "off" if disabled else "auto")
    monkeypatch.setenv(moe_schedule._PREFILL_ENV_NAME, "off" if disabled else "auto")
    monkeypatch.setenv(moe_schedule._R2_M4_BASELINE_ENV, "0")

    def device_name():
        assert not disabled, "disabled schedules must not probe the device"
        return "MTT S4000"

    monkeypatch.setattr(moe_schedule, "_device_name", device_name)

    class UnreadableShape:
        def __len__(self):
            pytest.fail("ineligible schedules must not inspect shapes")

    shapes = (UnreadableShape(), UnreadableShape())
    sentinel = object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    options = dict(
        is_marlin=True,
        block_shape=[128, 128],
        per_channel_quant=True,
        return_down_config=True,
    )
    assert wrapped(*shapes, 8, None, 64, **options) is sentinel
    assert calls == [((*shapes, 8, None, 64), options)]


def test_installed_wrapper_keeps_independent_m4_switch_live(monkeypatch):
    monkeypatch.setenv(moe_schedule._ENV_NAME, "off")
    monkeypatch.setenv(moe_schedule._PREFILL_ENV_NAME, "off")
    monkeypatch.setenv(moe_schedule._R2_M4_BASELINE_ENV, "0")
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    sentinel = object()
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(lambda *a, **kw: sentinel)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) is sentinel
    monkeypatch.setenv(moe_schedule._R2_M4_BASELINE_ENV, "1")
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) == moe_schedule._R2_M4_BASELINE_CONFIG
    monkeypatch.setenv(moe_schedule._R2_M4_BASELINE_ENV, "0")
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) is sentinel


def test_s5000_long_prefill_uses_tuned_schedule(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", raising=False)

    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return {"original": True}

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    for tokens in (2048, 8192):
        config, (down_config, max_block_m) = wrapped(
            W1_SHAPE,
            W2_SHAPE,
            8,
            None,
            tokens,
            return_down_config=True,
        )
        assert config == moe_schedule._S5000_PREFILL_CONFIG
        assert down_config == config
        assert down_config is not config
        assert max_block_m == 32
    assert calls == []


def test_s5000_exact_m16k_prefill_uses_bm64_schedule(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", raising=False)

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
        16384,
        return_down_config=True,
    )

    assert config == moe_schedule._S5000_PREFILL_M16K_CONFIG
    assert config["BLOCK_SIZE_M"] == 64
    assert down_config == config
    assert down_config is not config
    assert max_block_m == 64
    assert calls == []


def test_prefill_schedule_is_scoped_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", "off")

    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 8192) is sentinel
    monkeypatch.delenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE")
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 1024) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 16383) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 16385) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 8193) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, "fp8_w8a8", 8192) is sentinel
    assert wrapped((256, 1024, 2048), W2_SHAPE, 8, None, 8192) is sentinel


def test_exact_m16k_respects_prefill_disable(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", "off")

    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 16384) is sentinel


def test_schedule_can_be_disabled(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setenv("SGLANG_MUSA_MOE_DECODE_SCHEDULE", "off")

    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(original)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 64) is sentinel


def test_r2_m4_baseline_is_explicit_and_returns_independent_configs(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE", raising=False)
    sentinel = object()
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(lambda *a, **kw: sentinel)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) is sentinel
    monkeypatch.setenv("SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE", "1")
    expected = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1)
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) == expected
    up, down = wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4, return_down_config=True)
    assert up == expected and down == (None, None)
    up['BLOCK_SIZE_N'] = 64
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) == expected


def test_decode_accepts_canonical_intermediate_size_512(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_MOE_DECODE_SCHEDULE", raising=False)

    sentinel = object()
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(lambda *a, **kw: sentinel)
    w1_512 = (256, 1024, 2048)
    w2_512 = (256, 2048, 512)

    config, (down_config, max_block_m) = wrapped(
        w1_512, w2_512, 8, None, 64, return_down_config=True
    )

    assert config == moe_schedule._S5000_DECODE_CONFIG
    assert "enable_backend_opt" not in config
    assert down_config == config
    assert down_config is not config
    assert max_block_m == 32


def test_prefill_rejects_intermediate_size_512(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", raising=False)

    sentinel = object()
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(lambda *a, **kw: sentinel)
    assert wrapped((256, 1024, 2048), (256, 2048, 512), 8, None, 4096) is sentinel


def test_backend_opt_stays_on_the_decode_256_config_copy(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setattr(moe_schedule, "_backend_opt_enabled", lambda: True)
    monkeypatch.delenv("SGLANG_MUSA_MOE_DECODE_SCHEDULE", raising=False)
    monkeypatch.delenv("SGLANG_MUSA_MOE_PREFILL_SCHEDULE", raising=False)

    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(
        lambda *a, **kw: {"original": True}
    )

    decode = wrapped(W1_SHAPE, W2_SHAPE, 8, None, 64)
    assert decode["enable_backend_opt"] is True
    # Every branch returns a fresh copy of the module constant.
    assert decode is not moe_schedule._S5000_DECODE_CONFIG
    # Module-level constants must never be mutated by a call.
    assert "enable_backend_opt" not in moe_schedule._S5000_DECODE_CONFIG
    assert "enable_backend_opt" not in moe_schedule._S5000_PREFILL_CONFIG

    prefill = wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4096)
    assert "enable_backend_opt" not in prefill

    decode_512 = wrapped((256, 1024, 2048), (256, 2048, 512), 8, None, 64)
    assert "enable_backend_opt" not in decode_512


def test_r2_m4_restore_does_not_expand_shape_or_feature_contract(monkeypatch):
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.setenv("SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE", "1")
    sentinel = object()
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(lambda *a, **kw: sentinel)
    for batch in (1, 2, 3, 5, 8):
        assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, batch) is sentinel
    for kwargs in ({'is_marlin': True}, {'per_channel_quant': True}, {'block_shape': [128, 128]}):
        assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4, **kwargs) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 4, None, 4) is sentinel
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, 'float32', 4) is sentinel
    assert wrapped((256, 1024, 2048), (256, 2048, 512), 8, None, 4) is sentinel
    monkeypatch.setattr(moe_schedule, "_device_name", lambda: "MTT S4000")
    assert wrapped(W1_SHAPE, W2_SHAPE, 8, None, 4) is sentinel
