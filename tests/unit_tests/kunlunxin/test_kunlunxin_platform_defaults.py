# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Kunlunxin platform wiring and its dispatch configuration.

Two layers are covered:

  * ``sglang_fl/platform.py`` — the vendor branch that fills ServerArgs defaults
    and the FlagCX bootstrap rule, plus the OOT attention-backend map used as
    the CLI default.
  * ``sglang_fl/dispatch/config/kunlunxin.yaml`` — platform auto-detection and
    the resolved policy (deny the CUDA-compat ``cuda`` backend, pin ops whose
    FlagGems/klx kernel is broken, keep the documented ATen blacklist).
"""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import sglang_fl
from sglang_fl import platform as platform_module
from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
from sglang_fl.dispatch.backends.flagos.register_ops import (
    register_builtins as register_flagos,
)
from sglang_fl.dispatch.backends.reference.register_ops import (
    register_builtins as register_reference,
)
from sglang_fl.dispatch.backends.vendor.kunlunxin.kunlunxin import KunlunxinBackend
from sglang_fl.dispatch.backends.vendor.kunlunxin.register_ops import (
    register_builtins as register_kunlunxin,
)
from sglang_fl.dispatch.config import utils as config_utils
from sglang_fl.dispatch.manager import OpManager
from sglang_fl.dispatch.policy import SelectionPolicy, policy_context
from sglang_fl.dispatch.registry import OpRegistry


def _kunlunxin_platform():
    platform = object.__new__(platform_module.PlatformFL)
    platform._vendor_name = "kunlunxin"
    platform._device_type = "cuda"
    return platform


@pytest.fixture
def clean_fl_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SGLANG_FL_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)


def test_default_attention_backend_is_the_oot_backend():
    assert platform_module._ATTN_BACKEND_MAP["kunlunxin"] == "kunlunxin"
    assert _kunlunxin_platform().get_default_attention_backend() == "kunlunxin"


def test_server_args_defaults_disable_mla_backends_but_keep_cuda_graphs():
    args = SimpleNamespace(
        attention_backend=None,
        mm_attention_backend=None,
        disable_cuda_graph=True,
    )

    _kunlunxin_platform().apply_server_args_defaults(args)

    assert args.mm_attention_backend == "sdpa"
    assert args.disable_cuda_graph is False
    assert args.attention_backend == "kunlunxin"


def test_explicit_attention_backend_is_preserved():
    args = SimpleNamespace(
        attention_backend="torch_native",
        mm_attention_backend=None,
        disable_cuda_graph=False,
    )

    _kunlunxin_platform().apply_server_args_defaults(args)

    assert args.attention_backend == "torch_native"


def test_flagcx_bootstraps_on_the_native_c10d_backend():
    platform = _kunlunxin_platform()
    platform._dist_backend = "flagcx"

    # FlagCX is not a c10d backend: process groups still bootstrap on nccl.
    assert platform.get_torch_distributed_backend_str() == "nccl"
    assert platform.get_communicator_class().__name__ == "CommunicatorFL"


def test_dispatch_config_contents(clean_fl_env, monkeypatch):
    monkeypatch.setattr(config_utils, "get_platform_name", lambda: "kunlunxin")

    config = sglang_fl._build_config()

    # ``cuda`` is a candidate backend because torch reports CUDA-compat.
    assert config["deny_vendors"] == {"cuda"}
    assert config["prefer"] == "flagos"
    assert config["op_backends"]["topk"] == ["reference"]
    assert config["op_backends"]["gemma_rms_norm"] == ["reference"]
    blacklist = set(config["flagos_blacklist"])
    for op in ("mm", "bmm", "index_put_", "stack", "gelu", "softmax"):
        assert op in blacklist, op


def _manager_with_all_backends(monkeypatch, *, flagos_available):
    monkeypatch.setattr(FlagOSBackend, "is_available", lambda self: flagos_available)
    monkeypatch.setattr(FlagOSBackend, "is_op_available", lambda self, name: flagos_available)
    monkeypatch.setattr(KunlunxinBackend, "is_available", lambda self: True)
    monkeypatch.setattr(KunlunxinBackend, "_available", True, raising=False)

    registry = OpRegistry()
    register_flagos(registry)
    register_kunlunxin(registry)
    register_reference(registry)
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    return manager


def test_pinned_ops_bypass_flaggems_even_when_it_is_available(monkeypatch):
    manager = _manager_with_all_backends(monkeypatch, flagos_available=True)
    policy = SelectionPolicy.from_dict(
        prefer="flagos",
        per_op_order={"topk": ["reference"], "gemma_rms_norm": ["reference"]},
        deny_vendors={"cuda"},
        strict=False,
    )

    with policy_context(policy):
        for op_name in ("topk", "gemma_rms_norm"):
            fn = manager.resolve(op_name)
            assert manager._get_impl_id_for_fn(op_name, fn) == "reference.torch"
        # Unpinned ops still prefer FlagGems (the documented default).
        fn = manager.resolve("rms_norm")
        assert manager._get_impl_id_for_fn("rms_norm", fn) == "default.flagos"


def test_vendor_impls_serve_kunlunxin_when_flaggems_is_disabled(monkeypatch):
    manager = _manager_with_all_backends(monkeypatch, flagos_available=False)
    policy = SelectionPolicy.from_dict(
        prefer="flagos", deny_vendors={"cuda"}, strict=False
    )

    with policy_context(policy):
        for op_name in (
            "fused_moe",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        ):
            fn = manager.resolve(op_name)
            assert manager._get_impl_id_for_fn(op_name, fn) == "vendor.kunlunxin"
