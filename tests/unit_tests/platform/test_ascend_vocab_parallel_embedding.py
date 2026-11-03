# Copyright 2026 FlagOS Contributors
"""Boundary, fallback, and communication-contract tests for fused embedding."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import vocab_parallel_embedding as patch


def check_communication_contract_and_idempotence(scattered, attn_tp):
    import sglang.srt.layers.vocab_parallel_embedding as module

    events = []
    output = object()
    reduced = object()

    class FakeEmbedding:
        def forward(self, ids):
            events.append("fallback")
            return "fallback"

    @contextmanager
    def symmetric(group, disabled):
        assert group == "tp_group" and disabled is False
        events.append("allocate_enter")
        yield
        events.append("allocate_exit")

    def fused(*args):
        events.append("lookup")
        return output

    def reduce_tp(x):
        assert x is output
        events.append("attn_tp" if attn_tp else "tp")
        return reduced

    with (
        mock.patch.object(module, "VocabParallelEmbedding", FakeEmbedding),
        mock.patch.object(module, "use_symmetric_memory", symmetric),
        mock.patch.object(module, "get_tp_group", lambda: "tp_group"),
        mock.patch.object(module, "is_allocation_symmetric", lambda: True),
        mock.patch.object(module, "get_attn_tp_context", lambda: SimpleNamespace(input_scattered=scattered)),
        mock.patch.object(module, "attn_tp_all_reduce", reduce_tp),
        mock.patch.object(module, "tensor_model_parallel_all_reduce", reduce_tp),
        mock.patch.object(patch, "_supported", lambda *args: True),
        mock.patch.object(patch, "fused_vocab_embedding", fused),
    ):
        assert patch.patch_vocab_parallel_embedding()
        installed = FakeEmbedding.forward
        assert not patch.patch_vocab_parallel_embedding()
        assert FakeEmbedding.forward is installed
        layer = FakeEmbedding()
        layer.weight = "weight"
        layer.shard_indices = "shard"
        layer.use_attn_tp_group = attn_tp
        assert layer.forward("ids") is (output if scattered else reduced)
        assert events == ["allocate_enter", "lookup", "allocate_exit"] + ([] if scattered else ["attn_tp" if attn_tp else "tp"])
        with mock.patch.object(patch, "_supported", lambda *args: False):
            assert layer.forward("ids") == "fallback"


def check_cpu_and_quantized_fallback():
    class Unquantized:
        pass

    layer = SimpleNamespace(tp_size=2, quant_method=Unquantized(), weight=torch.empty(128, 32))
    assert not patch._supported(layer, torch.zeros(3, dtype=torch.int64), Unquantized)
    layer.quant_method = object()
    assert not patch._supported(layer, torch.zeros(3, dtype=torch.int64), Unquantized)
    layer.tp_size = 1
    assert not patch._supported(layer, torch.zeros(3, dtype=torch.int64), Unquantized)


@torch.inference_mode()
def check_exact_added_vocab_padding_and_changed_graph(dtype, hidden, id_dtype):
    shard = SimpleNamespace(org_vocab_start_index=100, org_vocab_end_index=200, num_org_vocab_padding=28, added_vocab_start_index=218, added_vocab_end_index=225)
    weight = torch.randn(160, hidden, device="npu", dtype=dtype)
    ids = torch.tensor([-1, 0, 99, 100, 199, 200, 217, 218, 224, 225, 2**30], device="npu", dtype=id_dtype)

    def expected():
        org = (ids >= 100) & (ids < 200)
        added = (ids >= 218) & (ids < 225)
        offset = 100 * org + 90 * added
        index = (org | added) * (ids - offset)
        return torch.nn.functional.embedding(index.long(), weight).masked_fill_(~(org | added).unsqueeze(-1), 0)

    for _ in range(3):
        actual = patch.fused_vocab_embedding(weight, ids, shard)
    assert torch.equal(actual, expected())
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = patch.fused_vocab_embedding(weight, ids, shard)
    ids.copy_(torch.tensor([224, 100, 199, 225, -1, 0, 218, 217, 200, 99, 2**30], device="npu", dtype=id_dtype))
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(actual, expected())


class TestAscendVocabParallelEmbedding(unittest.TestCase):
    def test_communication_contract_and_idempotence(self):
        for scattered in (False, True):
            for attn_tp in (False, True):
                with self.subTest(scattered=scattered, attn_tp=attn_tp):
                    check_communication_contract_and_idempotence(scattered, attn_tp)

    def test_cpu_and_quantized_fallback(self):
        check_cpu_and_quantized_fallback()

    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "Ascend required")
    def test_exact_boundaries_and_changed_graph(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for hidden in (2048, 5120, 8192):
                for id_dtype in (torch.int32, torch.int64):
                    with self.subTest(dtype=dtype, hidden=hidden, id_dtype=id_dtype):
                        check_exact_added_vocab_padding_and_changed_graph(dtype, hidden, id_dtype)

    @unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "Ascend required")
    def test_npu_supported_contract(self):
        class Unquantized:
            pass

        class CustomQuantized(Unquantized):
            pass

        weight = torch.empty(160, 2048, device="npu", dtype=torch.bfloat16)
        ids = torch.zeros(8, device="npu", dtype=torch.int64)
        layer = SimpleNamespace(tp_size=2, quant_method=Unquantized(), weight=weight,
                                shard_indices=SimpleNamespace(num_elements_padded=160))
        with torch.inference_mode():
            assert patch._supported(layer, ids, Unquantized)
            assert patch._supported(layer, ids.to(torch.int32), Unquantized)
            assert not patch._supported(layer, ids[::2], Unquantized)
            assert not patch._supported(layer, ids.to(torch.int16), Unquantized)
            layer.quant_method = CustomQuantized()
            assert not patch._supported(layer, ids, Unquantized)
            layer.quant_method = Unquantized()
            layer.weight = weight.T
            assert not patch._supported(layer, ids, Unquantized)
            layer.weight = weight.to(torch.int32)
            assert not patch._supported(layer, ids, Unquantized)
            layer.weight = weight
            import torch_npu
            assert torch_npu.get_npu_format(weight) == 2
            assert torch_npu.get_npu_format(ids) == 2
            # torch_npu exposes a setter-only config property in this version.
            previous_option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
            allow_internal = previous_option == b"enable"
            try:
                torch.npu.config.allow_internal_format = True
                layer.weight = torch_npu.npu_format_cast(weight, 29)
            finally:
                torch.npu.config.allow_internal_format = allow_internal
            assert torch_npu.get_npu_format(layer.weight) == 29
            assert not patch._supported(layer, ids, Unquantized)
            layer.weight = weight
        with torch.enable_grad():
            assert not patch._supported(layer, ids, Unquantized)


if __name__ == "__main__":
    unittest.main()
