# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads import env


@pytest.mark.parametrize("value", [None, "auto", "1", "true", "yes", "on"])
def test_default_on_controls_allow_automatic_selection(monkeypatch, value):
    name = "SGLANG_MUSA_M4_W13_BN64"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    assert env.enabled(name, default=True)


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "disabled", "typo", ""])
def test_explicit_disable_and_unknown_values_do_not_enable(monkeypatch, value):
    monkeypatch.setenv("MUSA_TEST_CONTROL", value)
    assert not env.enabled("MUSA_TEST_CONTROL", default=True)


def test_experiments_stay_default_off_without_changing_environment(monkeypatch):
    monkeypatch.delenv("MUSA_TEST_CONTROL", raising=False)
    assert not env.enabled("MUSA_TEST_CONTROL")
    monkeypatch.setenv("MUSA_TEST_CONTROL", "auto")
    assert not env.enabled("MUSA_TEST_CONTROL")
    assert env.enabled("MUSA_TEST_CONTROL", default=True)
    assert env.os.environ["MUSA_TEST_CONTROL"] == "auto"


@pytest.mark.parametrize("name,expected", [("MTT S5000", True), ("MTT S4000", False)])
def test_hardware_default_uses_runtime_device_identity(monkeypatch, name, expected):
    monkeypatch.setattr(
        torch, "musa", NS(get_device_name=lambda device: name), raising=False
    )
    assert env.is_s5000(NS(type="musa")) is expected
    assert not env.is_s5000(torch.device("cpu"))


def test_unavailable_device_query_does_not_enable_s5000_defaults(monkeypatch):
    monkeypatch.delattr(torch, "musa", raising=False)
    assert not env.is_s5000(NS(type="musa"))
