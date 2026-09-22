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

"""Regression tests for the independent SGLang 0.5.18 Ascend adaptation."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml


def test_ascend_chunk_disables_wheel_final_state_by_default(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.fla import (
        chunk_gated_delta_rule_ascend,
    )

    kernel_module = ModuleType("sgl_kernel_npu.fla.chunk")
    calls = []
    expected = object()
    kernel_module.chunk_gated_delta_rule_npu = lambda **kwargs: (
        calls.append(kwargs) or expected
    )
    monkeypatch.setitem(sys.modules, kernel_module.__name__, kernel_module)

    tensors = [object() for _ in range(5)]
    assert chunk_gated_delta_rule_ascend(*tensors, scale=0.5) is expected
    assert calls[0]["output_final_state"] is False


def test_ascend_moe_delegates_the_0518_dispatch_output_unchanged() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        fused_moe_ascend,
    )

    calls = []
    expected = object()
    layer = object()
    dispatch_output = SimpleNamespace(
        # AscendTPDispatchOutput has these fields and deliberately has no
        # pre-0.5.18 `topk_output` aggregate.
        hidden_states=object(),
        topk_weights=object(),
        topk_ids=object(),
    )
    method = SimpleNamespace(
        forward_npu=lambda actual_layer, actual_dispatch: (
            calls.append((actual_dispatch, actual_layer)) or expected
        )
    )

    assert fused_moe_ascend(method, layer, dispatch_output) is expected
    assert calls == [(dispatch_output, layer)]


def test_ascend_normalization_delegates_native_0518_fallbacks() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.normalization import (
        gemma_rms_norm_ascend,
        rms_norm_ascend,
    )

    calls = []
    expected = object()
    x = object()
    residual = object()
    norm = SimpleNamespace(
        forward_npu=lambda actual_x, actual_residual: (
            calls.append((actual_x, actual_residual)) or expected
        )
    )

    assert rms_norm_ascend(norm, x, residual) is expected
    assert gemma_rms_norm_ascend(norm, x, None) is expected
    assert calls == [(x, residual), (x, None)]


def test_ascend_topk_delegates_native_0518_routing() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.topk import topk_ascend

    calls = []
    expected = object()
    hidden_states = object()
    router_logits = object()
    non_padded = object()
    dispatch_info = object()
    topk = SimpleNamespace(
        forward_npu=lambda *args, **kwargs: calls.append((args, kwargs)) or expected
    )

    assert (
        topk_ascend(
            topk,
            hidden_states,
            router_logits,
            num_token_non_padded=non_padded,
            expert_location_dispatch_info=dispatch_info,
        )
        is expected
    )
    assert calls == [
        (
            (hidden_states, router_logits),
            {
                "num_token_non_padded": non_padded,
                "expert_location_dispatch_info": dispatch_info,
            },
        )
    ]


def test_ascend_qwen36_policy_uses_0518_native_contracts() -> None:
    config_path = (
        Path(__file__).parents[3] / "sglang_fl" / "dispatch" / "config" / "ascend.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backends = config["op_backends"]

    for op_name in ("topk", "gemma_rms_norm", "fused_moe", "chunk_gated_delta_rule"):
        assert backends[op_name][0] == "vendor"
    assert backends["silu_and_mul"][0] == "flagos"
    assert backends["mrotary_embedding"][0] == "flagos"
    assert "fused_recurrent_gated_delta_rule" not in backends
    assert "masked_scatter_" in config["flagos_blacklist"]
    assert "sum" not in config["flagos_blacklist"]


def test_ascend_entrypoints_do_not_export_removed_spec_v2_toggle() -> None:
    root = Path(__file__).parents[3]
    entrypoints = [
        root / "scripts" / "ascend" / "acceptance_common.sh",
        *sorted((root / "examples").glob("qwen3_6_*.py")),
    ]

    for entrypoint in entrypoints:
        assert "SGLANG_ENABLE_SPEC_V2" not in entrypoint.read_text(encoding="utf-8"), (
            f"{entrypoint.relative_to(root)} exports the toggle removed by SGLang 0.5.18"
        )


def test_ascend_single_node_entrypoints_default_gloo_to_loopback() -> None:
    root = Path(__file__).parents[3]
    examples = root / "examples"
    single_node_entrypoints = [
        examples / "qwen3_6_27b_concurrent.py",
        examples / "qwen3_6_27b_mtp_inference.py",
        examples / "qwen3_6_27b_offline_inference.py",
        examples / "qwen3_6_35b_a3b_concurrent.py",
        examples / "qwen3_6_35b_a3b_offline_inference.py",
    ]

    for entrypoint in single_node_entrypoints:
        source = entrypoint.read_text(encoding="utf-8")
        assert 'os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")' in source

    common = (root / "scripts" / "ascend" / "acceptance_common.sh").read_text(
        encoding="utf-8"
    )
    assert 'GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"' in common
    assert 'export GLOO_SOCKET_IFNAME="${interface}"' in common


def test_ascend_fla_patch_preserves_native_gdn_state_contract(monkeypatch) -> None:
    from sglang_fl.dispatch import fla_patch
    from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge

    native_chunk = object()
    native_recurrent = object()
    native_packed = object()
    chunk_module = SimpleNamespace(chunk_gated_delta_rule=native_chunk)
    recurrent_module = SimpleNamespace(
        fused_recurrent_gated_delta_rule=native_recurrent,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )
    gdn_module = SimpleNamespace(
        chunk_gated_delta_rule=native_chunk,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )

    modules = {
        "sglang.kernels.ops.attention.fla.chunk": chunk_module,
        "sglang.kernels.ops.attention.fla.fused_recurrent": recurrent_module,
        "sglang.srt.layers.attention.linear.kernels.gdn_triton": gdn_module,
    }
    monkeypatch.setattr(
        fla_patch.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(fla_patch, "_originals", {})

    originals = fla_patch.patch_fla_functions(
        excluded_ops={
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
        excluded_gdn_ops={
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
    )

    # The public chunk function keeps its SGLang `(output, None, h)` contract
    # through the bridge, while the internal NPU alias retains the wheel's
    # `(output, final_state, h)` contract used for cache writeback.
    assert chunk_module.chunk_gated_delta_rule is chunk_gated_delta_rule_bridge
    assert gdn_module.chunk_gated_delta_rule is native_chunk
    assert recurrent_module.fused_recurrent_gated_delta_rule is native_recurrent
    assert (
        recurrent_module.fused_recurrent_gated_delta_rule_packed_decode is native_packed
    )
    assert gdn_module.fused_recurrent_gated_delta_rule_packed_decode is native_packed
    assert originals == {"chunk_gated_delta_rule": native_chunk}


def test_ascend_fla_patch_first_gdn_import_keeps_npu_rebinding(monkeypatch) -> None:
    from sglang_fl.dispatch import fla_patch
    from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge

    native_public_chunk = object()
    native_npu_chunk = object()
    native_recurrent = object()
    native_packed = object()
    chunk_module = SimpleNamespace(chunk_gated_delta_rule=native_public_chunk)
    recurrent_module = SimpleNamespace(
        fused_recurrent_gated_delta_rule=native_recurrent,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )
    imported_gdn = []

    def import_module(name):
        if name == "sglang.kernels.ops.attention.fla.chunk":
            return chunk_module
        if name == "sglang.kernels.ops.attention.fla.fused_recurrent":
            return recurrent_module
        if name == "sglang.srt.layers.attention.linear.kernels.gdn_triton":
            # This mimics a first import on NPU: gdn_triton initially imports
            # the public name, then its is_npu() branch rebinds to the wheel.
            assert chunk_module.chunk_gated_delta_rule is chunk_gated_delta_rule_bridge
            module = SimpleNamespace(
                chunk_gated_delta_rule=native_npu_chunk,
                fused_recurrent_gated_delta_rule_packed_decode=native_packed,
            )
            imported_gdn.append(module)
            return module
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(fla_patch.importlib, "import_module", import_module)
    monkeypatch.setattr(fla_patch, "_originals", {})

    fla_patch.patch_fla_functions(
        excluded_ops={
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
        excluded_gdn_ops={
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
    )

    assert len(imported_gdn) == 1
    assert imported_gdn[0].chunk_gated_delta_rule is native_npu_chunk
    assert (
        imported_gdn[0].fused_recurrent_gated_delta_rule_packed_decode is native_packed
    )


def test_flagos_generic_recurrent_is_not_advertised_on_ascend(monkeypatch) -> None:
    import sglang_fl.utils as utils
    from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend

    backend = FlagOSBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(
        utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="ascend"),
    )

    assert backend.is_fused_recurrent_available() is False


def test_ascend_page_range_chunks_the_torch_npu_launch(monkeypatch) -> None:
    pytest.importorskip("sglang")
    import torch

    from sglang_fl.dispatch.backends.vendor.ascend import allocator

    calls = []
    native_arange = torch.arange

    def recording_arange(start, stop, **kwargs):
        calls.append((start, stop))
        kwargs["device"] = "cpu"
        return native_arange(start, stop, **kwargs)

    monkeypatch.setattr(allocator.torch, "arange", recording_arange)
    pages = allocator.build_ascend_page_range(
        131_075,
        device="npu",
        max_elements=65_535,
    )

    assert calls == [(1, 65_536), (65_536, 131_071), (131_071, 131_076)]
    assert all(stop - start <= 65_535 for start, stop in calls)
    assert pages.dtype == torch.int64
    assert pages.tolist() == list(range(1, 131_076))


def test_ascend_allocator_rejects_invalid_ranges() -> None:
    pytest.importorskip("sglang")
    from sglang_fl.dispatch.backends.vendor.ascend.allocator import (
        build_ascend_page_range,
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_ascend_page_range(-1, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        build_ascend_page_range(1, device="cpu", max_elements=0)
