# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise QSA dispatch against the actual decode-graph metadata contract."""

from types import SimpleNamespace

import pytest
import torch

from sglang_fl.models import qwen4_qsa


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="Requires MUSA Graph capture and replay",
)
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_qsa_graph_uses_live_request_pool_table(monkeypatch, rows):
    import flaggems_sglang
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnMetadata,
    )

    torch.manual_seed(42)
    device = "musa"
    table = torch.arange(1, rows * 4 * 64 + 1, device=device, dtype=torch.int32).view(
        rows * 4, 64
    )
    requests = torch.arange(rows, device=device, dtype=torch.int32) + rows
    keys = torch.randn(rows * 4 * 64 + 1, 1, 256, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    q = torch.randn(rows, 3, 256, device=device, dtype=keys.dtype)
    k = torch.randn(rows, 1, 256, device=device, dtype=keys.dtype)
    v = torch.randn_like(k)
    gate = torch.randn_like(q)
    selected = (
        torch.arange(33, device=device, dtype=torch.int32).expand(rows, -1).clone()
    )
    selected[:, -1] = -1
    locations = table[requests.long(), 31].clone()
    metadata = QwenSparseAttnMetadata(
        sequence_lengths=torch.full((rows,), 32, device=device, dtype=torch.int32),
        token_to_batch_idx=torch.arange(rows, device=device, dtype=torch.int32),
        # SGLang deliberately supplies this dummy table during decode capture.
        token_slot_table=torch.zeros(rows, 1, device=device, dtype=torch.int32),
        indexer_metadata=None,
        row_req_pool_indices=requests,
        is_cuda_graph=True,
    )
    cache = SimpleNamespace(key=lambda: keys, value=lambda: values)
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda _: cache)
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        _resolve_metadata=lambda _: metadata,
        _expand_block_indices=lambda indices, _: indices,
        _pad_extend_output=lambda output, _: output,
    )
    layer = SimpleNamespace(
        layer_id=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=3,
        head_dim=256,
        scaling=256**-0.5,
    )
    batch = SimpleNamespace(out_cache_loc=locations)
    monkeypatch.setattr(qwen4_qsa, "resolve_op", flaggems_sglang.get_op)

    def run():
        return qwen4_qsa._attention(
            None,
            backend,
            q,
            k,
            v,
            layer,
            batch,
            topk_indices=selected,
            qsa_output_gate=gate,
        )

    run()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        output = run()
    for phase in range(2):
        requests.copy_(torch.arange(rows, device=device).flip(0) + rows * (phase + 1))
        table.copy_(table.flip(1))
        locations.copy_(table[requests.long(), 31])
        q.normal_()
        k.normal_()
        v.normal_()
        gate.normal_()
        keys.normal_()
        values.normal_()
        reference_keys = keys.cpu()
        reference_values = values.cpu()
        reference_keys[locations.cpu().long()] = k.cpu()
        reference_values[locations.cpu().long()] = v.cpu()
        physical = table.cpu()[requests.cpu().long(), :32].long()
        gathered_k = reference_keys[physical, 0].float()
        gathered_v = reference_values[physical, 0].float()
        scores = (
            torch.einsum("bhd,bnd->bhn", q.cpu().float(), gathered_k) * layer.scaling
        )
        reference = torch.einsum("bhn,bnd->bhd", scores.softmax(-1), gathered_v).to(
            q.dtype
        ) * torch.sigmoid(gate.cpu())
        output.fill_(float("nan"))
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(output.cpu(), reference, atol=0.03, rtol=0.03)
        torch.testing.assert_close(keys.cpu(), reference_keys, atol=0, rtol=0)
        torch.testing.assert_close(values.cpu(), reference_values, atol=0, rtol=0)
