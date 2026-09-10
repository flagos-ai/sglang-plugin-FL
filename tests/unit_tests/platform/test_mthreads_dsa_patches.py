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

"""Unit coverage for the MUSA DSA OOT registration shim."""

import sys
import types


def test_registers_indexer_and_optional_kpool(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.mthreads import patch

    registrations = []

    class FakeBaseFusedOp:
        @classmethod
        def register_oot_forward(cls, op_cls, forward, backend):
            registrations.append((op_cls, forward, backend))

    class FakeIndexer:
        @staticmethod
        def forward_cuda():
            pass

    class FakeIndexerKPool:
        @staticmethod
        def forward_cuda():
            pass

    fused_op = types.ModuleType("sglang.kernels.fused_op")
    fused_op.BaseFusedOp = FakeBaseFusedOp
    indexer = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_indexer")
    indexer.Indexer = FakeIndexer
    kpool = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_indexer_kpool")
    kpool.IndexerKPool = FakeIndexerKPool
    monkeypatch.setitem(sys.modules, "sglang.kernels.fused_op", fused_op)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.dsa.dsa_indexer", indexer
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.dsa.dsa_indexer_kpool",
        kpool,
    )

    patch._register_dsa_indexer_forward()

    assert registrations == [
        (FakeIndexer, FakeIndexer.forward_cuda, "oot"),
        (FakeIndexerKPool, FakeIndexerKPool.forward_cuda, "oot"),
    ]


def test_missing_optional_kpool_does_not_block_indexer(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.mthreads import patch

    registrations = []

    class FakeBaseFusedOp:
        @classmethod
        def register_oot_forward(cls, op_cls, forward, backend):
            registrations.append((op_cls, forward, backend))

    class FakeIndexer:
        @staticmethod
        def forward_cuda():
            pass

    fused_op = types.ModuleType("sglang.kernels.fused_op")
    fused_op.BaseFusedOp = FakeBaseFusedOp
    indexer = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_indexer")
    indexer.Indexer = FakeIndexer
    monkeypatch.setitem(sys.modules, "sglang.kernels.fused_op", fused_op)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.dsa.dsa_indexer", indexer
    )
    monkeypatch.delitem(
        sys.modules,
        "sglang.srt.layers.attention.dsa.dsa_indexer_kpool",
        raising=False,
    )

    real_import = __import__

    def import_without_kpool(name, *args, **kwargs):
        if name == "sglang.srt.layers.attention.dsa.dsa_indexer_kpool":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_kpool)
    patch._register_dsa_indexer_forward()

    assert registrations == [(FakeIndexer, FakeIndexer.forward_cuda, "oot")]
