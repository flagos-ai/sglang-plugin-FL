# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_fl import profiling_evidence as evidence


def test_successful_registration_carries_exact_api_and_failed_calls_do_not(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_PROFILE_EVIDENCE_DIR", "/unused")
    recorded = []
    monkeypatch.setattr(evidence, "_write", recorded.append)
    monkeypatch.setattr(evidence, "triton_provider", lambda: {"name": "flagtree"})

    def add(fail=False):
        evidence.record_triton_launch(SimpleNamespace(name="exact_kernel"), add)
        if fail:
            raise RuntimeError("provider failed")
        return 42

    entry = ("add.Tensor", add)
    module = SimpleNamespace(
        _FULL_CONFIG=(entry,), FULL_CONFIG_BY_FUNC={"add": [entry]}
    )
    evidence.instrument_flaggems_registrations(module)
    fn = module._FULL_CONFIG[0][1]
    assert fn() == 42
    assert recorded[0]["aten_api"] == "aten::add"
    assert recorded[0]["component"] == "FlagGems"
    assert recorded[0]["kernel_name"] == "exact_kernel"
    recorded.clear()
    with pytest.raises(RuntimeError):
        fn(fail=True)
    assert not recorded
    assert evidence._active_aten.get() is None
    evidence.instrument_flaggems_registrations(module)
    assert module._FULL_CONFIG[0][1] is fn


def test_disabled_evidence_does_not_record(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_PROFILE_EVIDENCE_DIR", raising=False)
    monkeypatch.setattr(evidence, "_write", lambda _: pytest.fail("unexpected record"))
    evidence.record_triton_launch(SimpleNamespace(name="unused"), lambda: None)
