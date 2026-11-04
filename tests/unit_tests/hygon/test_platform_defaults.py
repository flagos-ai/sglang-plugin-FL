# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for HCU platform identity and documented dispatch defaults."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.config import utils
from sglang_fl.dispatch.policy import PolicyManager


@pytest.fixture
def platform_cls(monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.platforms"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    mixin = ModuleType("sglang.srt.platforms.device_mixin")
    mixin.PlatformEnum = type("PlatformEnum", (), {"OOT": "oot"})
    mixin.DeviceCapability = object
    interface = ModuleType("sglang.srt.platforms.interface")
    interface.SRTPlatform = object
    monkeypatch.setitem(sys.modules, mixin.__name__, mixin)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    path = Path(__file__).resolve().parents[3] / "sglang_fl/platform.py"
    spec = importlib.util.spec_from_file_location("_hcu_platform_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlatformFL


@pytest.mark.parametrize("hip,expected", [("6.3", "hip"), (None, "cuda")])
def test_hcu_retains_native_fused_op_fallback(monkeypatch, platform_cls, hip, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    platform = platform_cls.__new__(platform_cls)
    platform._vendor_name = "hygon"
    platform._device_type = "cuda"
    assert platform.get_dispatch_key_name() == expected
    assert platform.support_cuda_graph()
    assert not platform.support_piecewise_cuda_graph()
    platform._dist_backend = "flagcx"
    assert platform.get_torch_distributed_backend_str() == "nccl"


@pytest.fixture
def cuda_torch(monkeypatch):
    # Platform detection must not inherit an available MUSA/NPU backend from
    # the hardware on which these CPU regressions happen to run.
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", module)
    # CUDA-API platforms are split by FlagGems' DeviceDetector (thead vs nvidia);
    # keep the real detector of the host out of these CPU regressions.
    import sglang_fl.utils as fl_utils

    monkeypatch.setattr(fl_utils, "get_device_info", lambda: None)
    return module


def test_hcu_config_detected_before_cuda(cuda_torch):
    cuda_torch.__hcu_version__ = "3.6"
    assert utils.get_platform_name() == "hygon"


def test_ordinary_cuda_keeps_existing_config(cuda_torch):
    assert utils.get_platform_name() == "nvidia"


def test_documented_policy_and_explicit_override(monkeypatch):
    monkeypatch.setattr(utils, "get_platform_name", lambda: "hygon")
    for name in (
        "SGLANG_FL_CONFIG",
        "SGLANG_FL_PER_OP",
        "SGLANG_FL_PREFER",
        "SGLANG_FL_DENY_VENDORS",
        "SGLANG_FL_ALLOW_VENDORS",
    ):
        monkeypatch.delenv(name, raising=False)
    manager = PolicyManager.get_instance()
    manager.reset_global_policy()
    try:
        config = utils.get_effective_config()
        assert config["flagos_blacklist"] == ["count_nonzero", "cumsum", "layer_norm"]
        policy = manager.get_policy()
        assert not policy.is_vendor_allowed("cuda")
        assert policy.per_op_order_dict["gemma_rms_norm"] == ["reference"]
        assert policy.per_op_order_dict["fused_moe"] == ["vendor"]
        assert policy.per_op_order_dict[
            "fused_recurrent_gated_delta_rule_packed_decode"
        ] == ["vendor"]
        monkeypatch.setenv("SGLANG_FL_PER_OP", "gemma_rms_norm=vendor")
        manager.reset_global_policy()
        assert manager.get_policy().per_op_order_dict["gemma_rms_norm"] == ["vendor"]
    finally:
        manager.reset_global_policy()
