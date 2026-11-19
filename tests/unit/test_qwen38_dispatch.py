# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen3.8 dispatch and checkpoint-view ownership contracts."""

from collections import defaultdict
from importlib.util import find_spec
from types import SimpleNamespace
import pytest
import torch
from sglang_fl.models import qwen3_8, qwen4_hc, gdn_norm

pytestmark = pytest.mark.skipif(
    find_spec("sglang.srt.models.qwen4_exp") is None,
    reason="Qwen3.8 integration requires the pinned current SGLang runtime",
)


def _model(monkeypatch):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    hc = GatedResidual(
        HyperConnectionConfig(
            hc_count=4, hidden_size=16, hc_lowrank=8, hc_per_branch_norm=True
        )
    )
    from sglang_fl.models import qwen4_communication

    monkeypatch.setattr(qwen4_communication, "prepare", lambda: None)
    root = torch.nn.Module()
    root.add_module("hc", hc)
    monkeypatch.setattr(qwen4_hc, "resolve_op", lambda name: lambda *args: None)
    return root, hc


def test_checkpoint_aliases(monkeypatch):
    root, hc = _model(monkeypatch)
    keys = set(root.state_dict())
    qwen4_hc.prepare_model(None, root)
    assert set(root.state_dict()) == keys
    root.load_state_dict(
        {name: torch.randn_like(value) for name, value in root.state_dict().items()}
    )
    torch.testing.assert_close(
        hc._fl_down_inject_weight[:8], hc.input_mix_weight_down.weight, atol=0, rtol=0
    )
    torch.testing.assert_close(
        hc._fl_down_inject_weight[8:], hc.block_inject_weight.weight, atol=0, rtol=0
    )
    for weight in (hc.input_mix_weight_down.weight, hc.block_inject_weight.weight):
        assert (
            hc._fl_down_inject_weight.untyped_storage().data_ptr()
            == weight.untyped_storage().data_ptr()
        )


def test_missing_capability_precedes_weight_repacking(monkeypatch):
    root, hc = _model(monkeypatch)

    def missing(name):
        raise RuntimeError("Missing public operator")

    monkeypatch.setattr(qwen4_hc, "resolve_op", missing)
    with pytest.raises(RuntimeError, match="Missing public operator"):
        qwen4_hc.prepare_model(None, root)
    assert not hasattr(hc, "_fl_down_inject_weight")


def test_injection_reuse_and_no_execution_retry(monkeypatch):
    from flaggems_sglang.reference import get_reference

    root, hc = _model(monkeypatch)
    qwen4_hc.prepare_model(None, root)
    monkeypatch.setattr(qwen4_hc, "_eligible", lambda *args: True)
    calls = []

    def resolve(name):
        calls.append(name)
        return get_reference(name)

    monkeypatch.setattr(qwen4_hc, "resolve_op", resolve)

    def forbidden(*args):
        raise AssertionError("Unexpected native retry")

    mixed, residual = qwen4_hc._mix(
        forbidden, hc, torch.randn(7, 64, dtype=torch.bfloat16)
    )
    assert residual[2].shape == (7, 4)
    assert residual[2].stride(0) == 12
    monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
    assert qwen4_hc._combine(forbidden, hc, mixed, residual).shape == (7, 64)
    assert calls[-1] == "qwen4_hc_inject_combine"

    def broken(*args):
        raise ValueError("Kernel execution failure")

    monkeypatch.setattr(qwen4_hc, "resolve_op", lambda name: broken)
    with pytest.raises(ValueError, match="Kernel execution failure"):
        qwen4_hc._combine(forbidden, hc, mixed, residual)


def test_non_target_uses_original():
    marker = object()
    instance = SimpleNamespace(
        input_mix_weight_down=SimpleNamespace(weight=torch.empty(1)),
        input_mix_weight_up=SimpleNamespace(weight=torch.empty(1)),
    )
    assert qwen4_hc._mix(lambda *args: marker, instance, torch.empty(1)) is marker


def test_installation_atomic_retryable_idempotent(monkeypatch):
    import sglang_fl.utils
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.kernels.fused_op import BaseFusedOp

    monkeypatch.setattr(
        sglang_fl.utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="mthreads"),
    )
    monkeypatch.setattr(qwen3_8, "_registered", False)
    monkeypatch.setattr(HookRegistry, "_hooks", defaultdict(list))
    monkeypatch.setattr(BaseFusedOp, "_oot_forward_registry", {})
    monkeypatch.setattr(gdn_norm, "register_hooks", lambda: None)
    register = HookRegistry.register
    calls = []

    def fail_second(cls, target, hook, kind):
        calls.append(target)
        if len(calls) == 2:
            raise RuntimeError("Registration failure")
        register(target, hook, kind)

    monkeypatch.setattr(HookRegistry, "register", classmethod(fail_second))
    with pytest.raises(RuntimeError, match="Registration failure"):
        qwen3_8.register_hooks()
    assert not qwen3_8._registered and not HookRegistry._hooks
    assert not BaseFusedOp._oot_forward_registry
    monkeypatch.setattr(HookRegistry, "register", register)
    qwen3_8.register_hooks()
    count = sum(map(len, HookRegistry._hooks.values()))
    qwen3_8.register_hooks()
    assert sum(map(len, HookRegistry._hooks.values())) == count


def test_non_mthreads_does_not_patch(monkeypatch):
    import sglang_fl.utils

    monkeypatch.setattr(qwen3_8, "_registered", False)
    monkeypatch.setattr(
        sglang_fl.utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="nvidia"),
    )
    qwen3_8.register_hooks()
    assert not qwen3_8._registered


def test_qsa_non_target_uses_original():
    from sglang_fl.models import qwen4_qsa, qwen4_qsa_rope

    marker = object()
    original = lambda *args, **kwargs: marker
    instance = SimpleNamespace()
    assert qwen4_qsa._no_cuda_jit(original, instance) is marker
    assert qwen4_qsa._use_compress(original, instance, None) is marker
    assert qwen4_qsa._prefill(original, instance, *([None] * 7)) is marker
    assert qwen4_qsa._decode(original, instance, *([None] * 8)) is marker
    assert qwen4_qsa._forward(original, instance, *([None] * 4)) is marker
    assert qwen4_qsa_rope.apply_rope(original, instance, None, None) is marker


def test_qsa_attention_dispatch_metadata_and_no_retry(monkeypatch):
    from sglang_fl.models import qwen4_qsa

    key_cache = torch.randn(19, 1, 8)
    value_cache = torch.randn_like(key_cache)
    slots = torch.tensor([11, 3, 9], dtype=torch.int32)
    table = torch.tensor([[11, 3, 9], [1, 2, 4]], dtype=torch.int32)
    requests = torch.tensor([0, 1, 0], dtype=torch.int32)
    selected = torch.tensor([[0, -1], [0, 1], [1, 2]], dtype=torch.int32)
    metadata = SimpleNamespace(
        token_slot_table=table, token_to_batch_idx=requests, is_cuda_graph=False
    )
    cache = SimpleNamespace(key=lambda: key_cache, value=lambda: value_cache)
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda _: cache)
        ),
        _resolve_metadata=lambda _: metadata,
        _expand_block_indices=lambda indices, _: indices,
        _pad_extend_output=lambda out, _: out,
    )
    layer = SimpleNamespace(
        layer_id=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=2,
        head_dim=8,
        scaling=8**-0.5,
    )
    batch = SimpleNamespace(out_cache_loc=slots)
    q = torch.randn(3, 16)
    k = torch.randn(3, 8)
    v = torch.randn_like(k)
    gate = torch.randn(3, 32)[:, ::2]
    output = torch.randn(3, 2, 8)
    calls = []

    def store(k_cache, v_cache, locations, keys, values):
        calls.append("store")
        assert (
            k_cache.untyped_storage().data_ptr()
            == key_cache.untyped_storage().data_ptr()
        )
        assert (
            v_cache.untyped_storage().data_ptr()
            == value_cache.untyped_storage().data_ptr()
        )
        assert locations is slots
        torch.testing.assert_close(keys.flatten(1), k)
        torch.testing.assert_close(values.flatten(1), v)

    def attention(query, kc, vc, indices, pages, request_ids, scaling, gate):
        calls.append("attention")
        assert indices is selected and pages is table and request_ids is requests
        assert scaling == layer.scaling
        torch.testing.assert_close(query.flatten(1), q)
        assert gate.stride(-1) == 2
        return output

    ops = {"qsa_store_kv_cache_rows": store, "qsa_sparse_paged_attention": attention}
    monkeypatch.setattr(qwen4_qsa, "resolve_op", ops.__getitem__)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected native retry")

    result = qwen4_qsa._attention(
        forbidden,
        backend,
        q,
        k,
        v,
        layer,
        batch,
        topk_indices=selected,
        qsa_output_gate=gate,
    )
    assert result is output and calls == ["store", "attention"]

    def broken(*args, **kwargs):
        raise RuntimeError("Attention execution failure")

    ops["qsa_sparse_paged_attention"] = broken
    calls.clear()
    with pytest.raises(RuntimeError, match="Attention execution failure"):
        qwen4_qsa._attention(
            forbidden,
            backend,
            q,
            k,
            v,
            layer,
            batch,
            topk_indices=selected,
            qsa_output_gate=gate,
        )
    assert calls == ["store"]


def test_qsa_capability_checked_before_cache_mutation(monkeypatch):
    from sglang_fl.models import qwen4_qsa

    calls = []

    def resolve(name):
        calls.append(name)
        if name == "qsa_store_kv_cache_rows":
            raise RuntimeError("Missing cache operator")
        return lambda *args: calls.append("executed")

    monkeypatch.setattr(qwen4_qsa, "resolve_op", resolve)
    with pytest.raises(RuntimeError, match="Missing cache operator"):
        qwen4_qsa._attention(
            None, None, None, None, None, None, None, qsa_output_gate=None
        )
    assert calls == ["qsa_sparse_paged_attention", "qsa_store_kv_cache_rows"]


def test_older_sglang_without_model_is_unchanged(monkeypatch):
    import sglang_fl.utils

    monkeypatch.setattr(qwen3_8, "_registered", False)
    monkeypatch.setattr(
        sglang_fl.utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="mthreads"),
    )
    monkeypatch.setattr(qwen3_8, "find_spec", lambda _: None)
    qwen3_8.register_hooks()
    assert not qwen3_8._registered


def test_graph_replay_bound():
    from sglang_fl.musa_graph import qsa_context_supported

    assert qsa_context_supported([1, 2048], 2048)
    assert not qsa_context_supported([2049], 2048)
    assert not qsa_context_supported(None, 2048)
    assert qsa_context_supported(None, None)


def test_qsa_compression_ring_layout(monkeypatch):
    from sglang_fl.models import qwen4_qsa

    raw = torch.randn(16, 1, 128, dtype=torch.bfloat16)
    rope_positions = torch.arange(48, dtype=torch.int64).reshape(16, 3)
    compressed = torch.zeros(8, 1, 128, dtype=raw.dtype)
    groups = torch.tensor([[5, 7, 1, 3], [9, 4, 2, 6]], dtype=torch.int32)
    slots = torch.tensor([4, 1], dtype=torch.int32)
    rope = SimpleNamespace(
        rotary_dim=64,
        is_neox_style=True,
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        cos_sin_cache=torch.randn(64, 64),
    )
    pool = SimpleNamespace(
        qsa_rope_position_buffer=rope_positions,
        get_qsa_key_state_buffer=lambda _: raw,
        get_qsa_compressed_k_buffer=lambda _: compressed,
    )
    indexer = SimpleNamespace(
        _fl_qwen38=True,
        layer_id=2,
        index_head_dim=128,
        compress_ratio=4,
        rotary_emb=rope,
        k_layernorm=SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6),
    )
    calls = []
    monkeypatch.setattr(
        qwen4_qsa, "resolve_op", lambda name: lambda *args: calls.append((name, args))
    )
    qwen4_qsa._compress(None, indexer, pool, groups, slots)
    name, args = calls.pop()
    assert name == "qsa_compress_norm_mrope_store_groups"
    assert args[0].shape == (16, 1, 1, 128) and args[1] is groups
    assert (
        args[4] is slots
        and args[5].untyped_storage().data_ptr()
        == compressed.untyped_storage().data_ptr()
    )
    assert (
        args[7] is rope.cos_sin_cache and args[11] == (11, 11, 10) and args[12] is True
    )
    torch.testing.assert_close(args[2], torch.arange(2, dtype=torch.int32))
    torch.testing.assert_close(args[3], torch.full((2,), 3, dtype=torch.int32))
    assert (
        args[13].untyped_storage().data_ptr()
        == rope_positions.untyped_storage().data_ptr()
    )


def test_registered_qwen38_contracts_use_public_operators():
    import flaggems_sglang
    from sglang_fl.dispatch.registry import OpRegistry
    from sglang_fl.dispatch.backends.flagos.qwen4 import OP_NAMES, register_builtins

    registry = OpRegistry()
    register_builtins(registry)
    snapshot = registry.snapshot()
    for name in OP_NAMES:
        entries = snapshot.impls_by_op[name]
        assert len(entries) == 1
        assert entries[0].fn is flaggems_sglang.get_op(name)


def test_gdn_non_target_keeps_native_path():
    marker = object()
    instance = SimpleNamespace(forward_native=lambda *args: marker)
    assert gdn_norm.gated_rmsnorm_oot(instance, torch.empty(1)) is marker
