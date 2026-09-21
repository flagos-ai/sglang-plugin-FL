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

from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    fla._MATE_GDN_CACHE.clear()
    yield
    fla._MATE_GDN_CACHE.clear()


def _make_api(required):
    params = ", ".join(sorted(required))
    namespace = {}
    exec(f"def api({params}):\n    return 'ok'\n", namespace)
    return namespace["api"]


def _patch_import(monkeypatch, func):
    monkeypatch.setattr(fla, "importlib", SimpleNamespace(import_module=func))


def test_successful_load_is_cached(monkeypatch):
    calls = []
    api = _make_api(fla._MATE_GDN_REQUIRED_PARAMETERS)

    def fake_import(name):
        calls.append(name)
        return SimpleNamespace(gated_delta_rule_decode=api)

    _patch_import(monkeypatch, fake_import)

    assert fla._load_mate_gdn_decode() is api
    assert fla._load_mate_gdn_decode() is api
    assert calls == ["mate.gdn_decode"]
    assert fla._MATE_GDN_CACHE["decode"] is api


@pytest.mark.parametrize("exc_type", [ImportError, OSError, TypeError, ValueError])
def test_known_load_failure_is_not_retried(monkeypatch, exc_type):
    calls = []

    def fake_import(name):
        calls.append(name)
        raise exc_type("known failure")

    _patch_import(monkeypatch, fake_import)

    assert fla._load_mate_gdn_decode() is None
    assert fla._load_mate_gdn_decode() is None
    assert calls == ["mate.gdn_decode"]
    assert "decode" in fla._MATE_GDN_CACHE
    assert fla._MATE_GDN_CACHE["decode"] is None


def test_unreadable_signature_is_unavailable(monkeypatch):
    api = _make_api(fla._MATE_GDN_REQUIRED_PARAMETERS)
    _patch_import(
        monkeypatch,
        lambda name: SimpleNamespace(gated_delta_rule_decode=api),
    )
    monkeypatch.setattr(
        fla.inspect, "signature", lambda fn: (_ for _ in ()).throw(ValueError("bad"))
    )

    assert fla._load_mate_gdn_decode() is None
    assert fla._MATE_GDN_CACHE["decode"] is None


def test_unknown_exception_propagates_then_is_not_retried(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        raise RuntimeError("boom")

    _patch_import(monkeypatch, fake_import)

    with pytest.raises(RuntimeError, match="boom"):
        fla._load_mate_gdn_decode()
    # The attempt was recorded before importing, so the next call does not
    # reload and returns the not-yet-successful cache value.
    assert fla._load_mate_gdn_decode() is None
    assert calls == ["mate.gdn_decode"]


def test_missing_symbol_is_confirmed_unavailable(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        return SimpleNamespace()

    _patch_import(monkeypatch, fake_import)

    assert fla._load_mate_gdn_prefill() is None
    assert calls == ["mate.gdn_prefill"]
    assert fla._MATE_GDN_CACHE["prefill"] is None


def test_missing_required_parameter_is_unavailable(monkeypatch):
    def incomplete_api():
        return "ok"

    _patch_import(
        monkeypatch,
        lambda name: SimpleNamespace(chunk_gated_delta_rule=incomplete_api),
    )

    assert fla._load_mate_gdn_prefill() is None
    assert fla._MATE_GDN_CACHE["prefill"] is None


def test_decode_and_prefill_caches_are_independent(monkeypatch):
    decode_api = _make_api(fla._MATE_GDN_REQUIRED_PARAMETERS)
    prefill_api = _make_api(fla._MATE_GDN_PREFILL_REQUIRED_PARAMETERS)
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "mate.gdn_decode":
            raise ImportError("no decode")
        return SimpleNamespace(chunk_gated_delta_rule=prefill_api)

    _patch_import(monkeypatch, fake_import)

    assert fla._load_mate_gdn_decode() is None
    assert fla._load_mate_gdn_prefill() is prefill_api
    assert fla._MATE_GDN_CACHE["decode"] is None
    assert fla._MATE_GDN_CACHE["prefill"] is prefill_api
    # Decode failure does not suppress an independent prefill success.
    assert decode_api is not prefill_api


def test_prefill_inherits_decode_switch(monkeypatch):
    monkeypatch.delenv("SGLANG_MUSA_MATE_GDN", raising=False)
    monkeypatch.delenv("SGLANG_MUSA_MATE_GDN_PREFILL", raising=False)
    assert fla._mate_gdn_prefill_enabled()

    monkeypatch.setenv("SGLANG_MUSA_MATE_GDN", "off")
    assert not fla._mate_gdn_prefill_enabled()

    monkeypatch.setenv("SGLANG_MUSA_MATE_GDN_PREFILL", "on")
    assert fla._mate_gdn_prefill_enabled()


def test_disabled_call_does_not_pollute_loader_cache(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSA_MATE_GDN", "off")
    monkeypatch.setattr(
        fla, "_original", lambda name: (lambda **kwargs: "original")
    )

    result = fla.fused_recurrent_gated_delta_rule_packed_decode_musa(
        mixed_qkv=None,
        a=None,
        b=None,
        A_log=None,
        dt_bias=None,
        scale=1.0,
        initial_state=None,
        out=None,
        ssm_state_indices=None,
    )

    assert result == "original"
    assert fla._MATE_GDN_CACHE == {}
