# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    fa3_graph_metadata as patch,
)


def _backend_classes():
    class Parent:
        def init_forward_metadata_capture_cuda_graph(
            self,
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        ):
            self.calls.append(
                (
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )
            )
            if self.error is not None:
                raise self.error
            # Parent creates metadata; the wrapper must operate on this object.
            self.forward_metadata = SimpleNamespace(
                max_seq_len_k=1,
                page_table=self.page_table,
                cache_seqlens_int32=seq_lens,
            )
            return self.sentinel

    class Musa(Parent):
        def __init__(self):
            self.calls = []
            self.error = None
            self.sentinel = object()
            self.use_mla = False
            self.page_size = 64
            self.page_table = SimpleNamespace(shape=(64, 4096))

    return Parent, Musa


@pytest.mark.parametrize("keywords", [False, True])
def test_capture_updates_capacity_after_parent_and_preserves_inputs(keywords):
    parent, musa = _backend_classes()
    original = parent.init_forward_metadata_capture_cuda_graph
    assert patch._patch_backend(musa)
    obj = musa()
    seq_lens, req_indices = object(), object()
    mode = SimpleNamespace(is_decode=lambda: True)
    args = (64, 64, req_indices, seq_lens, None, mode, None)
    if keywords:
        result = obj.init_forward_metadata_capture_cuda_graph(
            **dict(zip(patch._PARAMETERS[1:], args))
        )
    else:
        result = obj.init_forward_metadata_capture_cuda_graph(*args)
    assert result is obj.sentinel
    assert obj.calls == [args]
    assert obj.forward_metadata.max_seq_len_k == 262144
    assert obj.forward_metadata.page_table is obj.page_table
    assert obj.forward_metadata.cache_seqlens_int32 is seq_lens
    assert parent.init_forward_metadata_capture_cuda_graph is original


@pytest.mark.parametrize("guard", ["mla", "extend", "idle", "spec", "no_pages"])
def test_other_capture_modes_retain_parent_metadata(guard):
    _, musa = _backend_classes()
    assert patch._patch_backend(musa)
    obj = musa()
    obj.use_mla = guard == "mla"
    if guard == "no_pages":
        obj.page_table = None
    mode = SimpleNamespace(is_decode=lambda: guard not in ("extend", "idle"))
    obj.init_forward_metadata_capture_cuda_graph(
        4,
        4,
        object(),
        object(),
        None,
        mode,
        object() if guard == "spec" else None,
    )
    assert obj.forward_metadata.max_seq_len_k == 1
    assert len(obj.calls) == 1


def test_each_capture_uses_its_own_page_capacity():
    _, musa = _backend_classes()
    assert patch._patch_backend(musa)
    obj = musa()
    for columns, page_size in ((72, 64), (8192, 32)):
        obj.page_table = SimpleNamespace(shape=(4, columns))
        obj.page_size = page_size
        obj.init_forward_metadata_capture_cuda_graph(
            4,
            4,
            object(),
            object(),
            None,
            SimpleNamespace(is_decode=lambda: True),
            None,
        )
        assert obj.forward_metadata.max_seq_len_k == columns * page_size


def test_idempotence_and_original_exception():
    _, musa = _backend_classes()
    assert patch._patch_backend(musa)
    wrapped = musa.init_forward_metadata_capture_cuda_graph
    assert patch._patch_backend(musa)
    assert musa.init_forward_metadata_capture_cuda_graph is wrapped
    obj = musa()
    obj.error = RuntimeError("parent failed")
    with pytest.raises(RuntimeError, match="parent failed") as exc:
        obj.init_forward_metadata_capture_cuda_graph(
            4,
            4,
            object(),
            object(),
            None,
            SimpleNamespace(is_decode=lambda: True),
            None,
        )
    assert exc.value is obj.error
    assert len(obj.calls) == 1
    assert not hasattr(obj, "forward_metadata")


def test_skip_backend_with_existing_override():
    parent, musa = _backend_classes()
    musa.init_forward_metadata_capture_cuda_graph = (
        parent.init_forward_metadata_capture_cuda_graph
    )
    original = musa.init_forward_metadata_capture_cuda_graph
    assert not patch._patch_backend(musa)
    assert musa.init_forward_metadata_capture_cuda_graph is original


def test_skip_changed_signature_and_missing_method():
    class Parent:
        def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
            pass

    class Musa(Parent):
        pass

    assert not patch._patch_backend(Musa)
    assert patch._METHOD not in vars(Musa)
    assert not patch._patch_backend(type("Missing", (), {}))


def test_public_installer_and_explicit_opt_out(monkeypatch):
    _, musa = _backend_classes()
    imports = []

    def load(name):
        imports.append(name)
        return SimpleNamespace(MusaFlashAttentionBackend=musa)

    monkeypatch.setattr(patch.importlib, "import_module", load)
    monkeypatch.setenv("SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY", "0")
    assert not patch.apply_musa_fa3_graph_metadata_patch()
    assert not imports
    monkeypatch.delenv("SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY")
    assert patch.apply_musa_fa3_graph_metadata_patch()
    assert imports == [
        "sglang.srt.hardware_backend.musa.attention.flashattention_backend"
    ]


def test_public_installer_unavailable_backend(monkeypatch):
    def unavailable(name):
        raise ImportError("not a MUSA installation")

    monkeypatch.delenv("SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY", raising=False)
    monkeypatch.setattr(patch.importlib, "import_module", unavailable)
    assert not patch.apply_musa_fa3_graph_metadata_patch()
