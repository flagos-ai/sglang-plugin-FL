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

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import fmha_schedule

CURRENT_CONFIG = (192, 64, 1, 1, 256, 256, 3, 3, False)


def _original(*args, **kwargs):
    return CURRENT_CONFIG


def test_s5000_8k_prefill_enables_pack_gqa(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)

    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(_original)
    tuned = wrapped(8192, 8, 256, 256, 2)

    assert tuned[:-1] == CURRENT_CONFIG[:-1]
    assert tuned[-1] is True


def test_fmha_schedule_is_narrow_and_respects_explicit_choice(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(_original)

    assert wrapped(4096, 8, 256, 256, 2) == CURRENT_CONFIG
    assert wrapped(8192, 4, 256, 256, 2) == CURRENT_CONFIG
    assert wrapped(8192, 8, 128, 128, 2) == CURRENT_CONFIG
    assert wrapped(8192, 8, 256, 256, 2, False) == CURRENT_CONFIG
    assert wrapped(8192, 8, 256, 256, 2, None, True) == CURRENT_CONFIG

    monkeypatch.setenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", "off")
    assert wrapped(8192, 8, 256, 256, 2) == CURRENT_CONFIG


def test_apply_patches_all_mate_aliases_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)
    utils = SimpleNamespace(_get_fwd_kernel_config=_original)
    fwd = SimpleNamespace(_get_fwd_kernel_config=_original)
    metadata = SimpleNamespace(_get_metadata_kernel_config=_original)

    def import_module(name):
        if name == "mate.jit.attention.fmha.fmha_utils":
            return utils
        if name == "mate.jit.attention.fmha.fmha_fwd":
            return fwd
        return metadata

    monkeypatch.setattr(fmha_schedule.importlib, "import_module", import_module)

    assert fmha_schedule.apply_musa_fmha_schedule_patch()
    assert utils._get_fwd_kernel_config is fwd._get_fwd_kernel_config
    assert utils._get_fwd_kernel_config is metadata._get_metadata_kernel_config
    assert getattr(utils._get_fwd_kernel_config, fmha_schedule._PATCH_MARKER)

    # Applying again must not re-wrap the already-patched aliases.
    assert fmha_schedule.apply_musa_fmha_schedule_patch()
    assert getattr(utils._get_fwd_kernel_config, fmha_schedule._PATCH_MARKER)


def test_legacy_selector_keeps_its_eight_argument_contract(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)
    calls = []

    def legacy(m, ratio, dim, dim_v, size, pack=None, qv=False, fp8=False):
        calls.append((m, ratio, dim, dim_v, size, pack, qv, fp8))
        return CURRENT_CONFIG

    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(legacy)
    assert wrapped(8192, 8, 256, 256, 2)[-1] is True
    assert calls == [(8192, 8, 256, 256, 2, None, False, False)]


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("high_pressure", [False, True])
def test_mate027_argument_is_forwarded_and_high_pressure_is_untuned(
    monkeypatch, keyword, high_pressure
):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)
    calls = []
    config = CURRENT_CONFIG[:-1] + (0, False)

    def upgraded(
        m,
        ratio,
        dim,
        dim_v,
        size,
        pack=None,
        qv=False,
        fp8=False,
        is_high_regpressure=False,
    ):
        calls.append(is_high_regpressure)
        return config

    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(upgraded)
    args = (8192, 8, 256, 256, 2, None, False, False)
    if keyword:
        result = wrapped(*args, is_high_regpressure=high_pressure)
    else:
        result = wrapped(*args, high_pressure)
    assert calls == [high_pressure]
    assert result[:-1] == config[:-1]
    assert result[-1] is (not high_pressure)
