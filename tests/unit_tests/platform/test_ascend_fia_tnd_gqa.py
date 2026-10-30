# Copyright 2026 FlagOS Contributors
"""FIA dispatch contracts: real NPU views, mocked attention and KV writes."""

import inspect
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import fia_tnd_gqa as patch


def _stock(
    self,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache=True,
    q_rope=None,
    k_rope=None,
    topk_indices=None,
    sinks=None,
    slopes=None,
):
    return "STOCK"


@pytest.fixture
def backend_class(monkeypatch):
    module = pytest.importorskip(
        "sglang.srt.hardware_backend.npu.attention.ascend_backend"
    )
    cls = module.AscendAttnBackend
    monkeypatch.setattr(cls, "forward_extend", _stock)
    monkeypatch.setenv("SGLANG_FL_FIA_TND_GQA", "1")
    return cls


def test_default_off_and_cpu_metadata(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_FIA_TND_GQA", raising=False)
    assert not patch.patch_fia_tnd_gqa()
    for value, expected in (
        (None, None),
        ([1, "2"], [1, 2]),
        (torch.tensor([3, 4]), [3, 4]),
        (torch.tensor([5], device="meta"), None),
        (object(), None),
    ):
        assert patch._cpu_ints(value) == expected


def test_signature_guard_and_idempotence(backend_class, monkeypatch):
    def wrong(self, q):
        pass

    monkeypatch.setattr(backend_class, "forward_extend", wrong)
    assert not patch.patch_fia_tnd_gqa()
    assert backend_class.forward_extend is wrong
    monkeypatch.setattr(backend_class, "forward_extend", _stock)
    with monkeypatch.context() as scoped:
        scoped.setattr(_stock, "__signature__", inspect.signature(wrong), raising=False)
        assert not patch.patch_fia_tnd_gqa()
        assert backend_class.forward_extend is _stock
    assert patch.patch_fia_tnd_gqa()
    wrapped = backend_class.forward_extend
    assert getattr(wrapped, patch._MARKER)
    assert not patch.patch_fia_tnd_gqa()
    assert backend_class.forward_extend is wrapped


def test_packed_v_dispatch_and_fallbacks(backend_class, monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    def tensor(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, device="npu", dtype=dtype)

    def changed(obj, **values):
        return NS(**dict(vars(obj), **values))

    q, k = tensor(16384, 3072), tensor(16384, 2, 256)
    v = tensor(16384, 7168)[:, -512:].view(16384, 2, 256)
    assert v.stride() == (7168, 256, 1)
    backend = NS(
        use_fia=True,
        use_mla=False,
        use_alibi=False,
        attn_cp_size=1,
        is_dllm_model=False,
        fia_mask=tensor(2048, 2048, dtype=torch.bool),
    )
    layer = NS(
        is_cross_attention=False,
        attn_type=NS(name="DECODER"),
        sliding_window_size=-1,
        logit_cap=0,
        tp_q_head_num=12,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=256,
        v_head_dim=256,
        scaling=0.0625,
    )
    kv = mock.Mock()
    batch = NS(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        seq_lens_cpu=[16384],
        extend_seq_lens_cpu=[16384],
        extend_prefix_lens_cpu=[0],
        encoder_lens=None,
        attn_cp_metadata=None,
        out_cache_loc=tensor(1, dtype=torch.int64),
        token_to_kv_pool=kv,
    )
    events = []
    kv.set_kv_buffer.side_effect = lambda *args: events.append("kv")

    def attention(q_, k_, v_, **kwargs):
        events.append("attention")
        return (q_,)

    fia = mock.Mock(side_effect=attention)
    monkeypatch.setattr(
        torch.ops.npu, "npu_fused_infer_attention_score", fia, raising=False
    )
    assert patch.patch_fia_tnd_gqa()
    inputs = dict(self=backend, q=q, k=k, v=v, layer=layer, forward_batch=batch)

    def call(**overrides):
        return backend_class.forward_extend(**dict(inputs, **overrides))

    result = call()
    assert result.shape == (16384, 3072) and result.is_contiguous()
    assert events == ["kv", "attention"]
    kv.set_kv_buffer.assert_called_once_with(layer, batch.out_cache_loc, k, v)
    for actual, source in zip(fia.call_args.args, (q, k, v)):
        assert actual.data_ptr() == source.data_ptr()
    assert fia.call_args.args[2].stride() == (7168, 256, 1)
    assert fia.call_args.kwargs == dict(
        num_heads=12,
        num_key_value_heads=2,
        input_layout="TND",
        atten_mask=backend.fia_mask,
        sparse_mode=3,
        scale=layer.scaling,
        next_tokens=0,
        actual_seq_lengths=[16384],
        actual_seq_lengths_kv=[16384],
    )
    fallbacks = [
        dict(v=tensor(16384, 4, 256)[:, :2]),
        dict(q=tensor(16384, 3073)[:, :3072]),
        dict(k=tensor(16384, 2, 257)[:, :, :256]),
        dict(forward_batch=changed(batch, seq_lens_cpu=[16385])),
        dict(
            forward_batch=changed(
                batch, extend_seq_lens_cpu=tensor(1, dtype=torch.int32)
            )
        ),
        dict(self=changed(backend, fia_mask=tensor(1024, 1024, dtype=torch.bool))),
        dict(self=changed(backend, fia_mask=tensor(2048, 2048, dtype=torch.float32))),
        dict(layer=changed(layer, tp_k_head_num=4)),
        dict(forward_batch=changed(batch, batch_size=2)),
        dict(save_kv_cache=False),
        dict(q=None),
    ]
    fallbacks.extend(
        {key: object()}
        for key in ("q_rope", "k_rope", "topk_indices", "sinks", "slopes")
    )
    for overrides in fallbacks:
        assert call(**overrides) == "STOCK"
    assert fia.call_count == kv.set_kv_buffer.call_count == 1
    # Preserve the existing runtime disable switch and never retry after KV writes.
    monkeypatch.setenv("SGLANG_FL_FIA_TND_GQA", "0")
    assert call() == "STOCK"
    monkeypatch.setenv("SGLANG_FL_FIA_TND_GQA", "1")
    fia.side_effect = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="device failure"):
        call()
    assert fia.call_count == kv.set_kv_buffer.call_count == 2
