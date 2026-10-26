# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os
import importlib
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.iluvatar.runtime_config import (
    apply_iluvatar_runtime_defaults,
)


def test_empty_defaults_keep_decode_graphs():
    args = SimpleNamespace(disable_cuda_graph=False)
    apply_iluvatar_runtime_defaults(args)
    assert args.attention_backend == "triton"
    assert args.sampling_backend == "pytorch"
    assert args.mm_attention_backend == "triton_attn"
    assert args.disable_custom_all_reduce
    assert args.disable_piecewise_cuda_graph
    assert not args.disable_cuda_graph


def test_explicit_backends_are_preserved():
    args = SimpleNamespace(
        attention_backend="torch_native",
        sampling_backend="pytorch",
        mm_attention_backend="sdpa",
    )
    apply_iluvatar_runtime_defaults(args)
    assert args.attention_backend == "torch_native"
    assert args.sampling_backend == "pytorch"
    assert args.mm_attention_backend == "sdpa"


def test_empty_policy_resolves_without_vendor_kernels(monkeypatch):
    import sglang_fl
    from sglang_fl.dispatch.config import utils as config_utils
    from sglang_fl.dispatch.registry import OpRegistry
    from sglang_fl.dispatch.manager import OpManager
    from sglang_fl.dispatch.policy import SelectionPolicy, policy_context
    from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
    from sglang_fl.dispatch.backends.flagos.register_ops import register_builtins
    from sglang_fl.dispatch.backends.vendor.iluvatar.iluvatar import IluvatarBackend
    from sglang_fl.dispatch.backends.vendor.iluvatar.register_ops import (
        register_builtins as register_iluvatar,
    )
    from sglang_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )

    for key in list(os.environ):
        if key.startswith("SGLANG_FL_"):
            monkeypatch.delenv(key)
    # Hardware detection precedes SGLANG_FL_PLATFORM on real CUDA/MUSA hosts.
    # Isolate only that detection while exercising the real Iluvatar YAML.
    monkeypatch.setattr(config_utils, "get_platform_name", lambda: "iluvatar")
    config = sglang_fl._build_config()
    assert config["deny_vendors"] == {"cuda"}
    assert config["flagos_blacklist"] == ["max", "masked_scatter_", "mul_"]
    assert config["op_backends"]["chunk_gated_delta_rule"] == [
        "vendor:iluvatar",
        "flagos",
    ]
    # The real operator implementations and their shape/state contracts are
    # tested separately. Here the kernel capability probes represent a fully
    # installed FlagGems release, so policy wiring can run without a GPU.
    monkeypatch.setattr(FlagOSBackend, "is_available", lambda self: True)
    monkeypatch.setattr(FlagOSBackend, "is_op_available", lambda self, name: True)
    monkeypatch.setattr(IluvatarBackend, "is_available", lambda self: True)
    registry = OpRegistry()
    register_builtins(registry)
    register_iluvatar(registry)
    register_reference(registry)
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    policy = SelectionPolicy.from_dict(
        prefer=config["prefer"],
        per_op_order=config["op_backends"],
        deny_vendors=config["deny_vendors"],
    )
    with policy_context(policy):
        fn = manager.resolve("gemma_rms_norm")
        assert manager._get_impl_id_for_fn("gemma_rms_norm", fn) == "reference.torch"
        for name in (
            "fused_moe",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        ):
            fn = manager.resolve(name)
            assert manager._get_impl_id_for_fn(name, fn) == "vendor.iluvatar"


def test_required_vendor_overlay_import_errors_propagate(monkeypatch):
    import sglang_fl
    import sglang_fl.utils

    monkeypatch.setattr(
        sglang_fl.utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="iluvatar"),
    )
    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name.endswith(".iluvatar.patch"):
            raise ImportError("CoreX dependency missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    with pytest.raises(ImportError, match="CoreX dependency missing"):
        sglang_fl._apply_vendor_patches()
