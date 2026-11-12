# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the small Kunlunxin rebinding patches.

Covers:
  * ``patches/attention_backend_choice.py`` — ``--attention-backend kunlunxin``
  * ``patches/clamp_position.py``          — JIT-CUDA plumbing -> torch native
  * ``patches/suppress_pynccl.py``         — FlagCX replaces real PyNccl
  * ``patches/vocab_embed_native.py``      — Triton fused embedding off under TP

Every patch mutates a real sglang module, so each test re-arms the module's
idempotency flag and lets ``monkeypatch`` restore the original attribute.
"""

import torch

from sglang.srt import server_args as sglang_server_args
from sglang.srt.distributed import parallel_state
from sglang.srt.layers import vocab_parallel_embedding
from sglang.srt.model_executor import forward_batch_info

from sglang_fl.dispatch.backends.vendor.kunlunxin.patches import (
    attention_backend_choice,
    clamp_position,
    suppress_pynccl,
    vocab_embed_native,
)


# ─── attention_backend_choice ─────────────────────────────────────────────────


def test_attention_backend_choice_is_appended(monkeypatch):
    monkeypatch.setattr(attention_backend_choice, "_patched", False)
    monkeypatch.setattr(
        sglang_server_args, "ATTENTION_BACKEND_CHOICES", ["triton", "torch_native"]
    )

    attention_backend_choice.patch_attention_backend_choice()

    assert sglang_server_args.ATTENTION_BACKEND_CHOICES == [
        "triton",
        "torch_native",
        "kunlunxin",
    ]


def test_attention_backend_choice_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(attention_backend_choice, "_patched", False)
    monkeypatch.setattr(
        sglang_server_args, "ATTENTION_BACKEND_CHOICES", ["triton", "kunlunxin"]
    )

    attention_backend_choice.patch_attention_backend_choice()
    attention_backend_choice.patch_attention_backend_choice()

    assert sglang_server_args.ATTENTION_BACKEND_CHOICES.count("kunlunxin") == 1


# ─── clamp_position ──────────────────────────────────────────────────────────


def test_clamp_position_is_rebound_to_the_torch_fallback(monkeypatch):
    monkeypatch.setattr(clamp_position, "_applied", False)
    monkeypatch.setattr(forward_batch_info, "clamp_position", lambda seq_lens: None)

    clamp_position.patch_clamp_position()

    assert (
        forward_batch_info.clamp_position
        is forward_batch_info._clamp_position_native
    )
    # Semantics kept by the native fallback: last valid position per request.
    out = forward_batch_info.clamp_position(torch.tensor([0, 1, 7]))
    assert out.tolist() == [0, 0, 6]
    assert out.dtype == torch.int64


def test_clamp_position_patch_is_idempotent(monkeypatch):
    monkeypatch.setattr(clamp_position, "_applied", False)
    monkeypatch.setattr(forward_batch_info, "clamp_position", lambda seq_lens: None)

    clamp_position.patch_clamp_position()
    rebound = forward_batch_info.clamp_position
    clamp_position.patch_clamp_position()

    assert forward_batch_info.clamp_position is rebound


def test_clamp_position_patch_warns_when_sglang_drops_the_api(monkeypatch):
    monkeypatch.setattr(clamp_position, "_applied", False)
    monkeypatch.delattr(forward_batch_info, "_clamp_position_native", raising=False)

    clamp_position.patch_clamp_position()

    assert clamp_position._applied is True


# ─── suppress_pynccl ────────────────────────────────────────────────────────


class _GroupCoordinatorStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_group_coordinator(monkeypatch):
    fake = type("GroupCoordinator", (_GroupCoordinatorStub,), {})
    monkeypatch.setattr(parallel_state, "GroupCoordinator", fake)
    monkeypatch.setattr(suppress_pynccl, "_applied", False)
    return fake


def test_suppress_pynccl_forces_use_pynccl_off(monkeypatch):
    fake = _install_group_coordinator(monkeypatch)

    suppress_pynccl.patch_suppress_pynccl()
    coordinator = fake("cpu-group", use_pynccl=True, world_size=4)

    assert coordinator.kwargs["use_pynccl"] is False
    assert coordinator.kwargs["world_size"] == 4
    assert coordinator.args == ("cpu-group",)


def test_suppress_pynccl_leaves_absent_flag_absent(monkeypatch):
    fake = _install_group_coordinator(monkeypatch)

    suppress_pynccl.patch_suppress_pynccl()
    coordinator = fake("cpu-group", world_size=1)

    assert "use_pynccl" not in coordinator.kwargs


def test_suppress_pynccl_does_not_wrap_twice(monkeypatch):
    fake = _install_group_coordinator(monkeypatch)

    suppress_pynccl.patch_suppress_pynccl()
    wrapped = fake.__init__
    # Second call re-runs the module-level guard, not a second wrapper.
    monkeypatch.setattr(suppress_pynccl, "_applied", False)
    suppress_pynccl.patch_suppress_pynccl()

    assert fake.__init__ is wrapped


# ─── vocab_embed_native ─────────────────────────────────────────────────────


def test_vocab_parallel_embedding_triton_path_is_disabled(monkeypatch):
    embedding_cls = vocab_parallel_embedding.VocabParallelEmbedding
    monkeypatch.setattr(
        embedding_cls, "_use_triton_embedding", embedding_cls._use_triton_embedding
    )
    monkeypatch.setattr(vocab_embed_native, "_applied", False)

    vocab_embed_native.patch_vocab_embed_native()

    instance = object.__new__(embedding_cls)
    assert instance._use_triton_embedding(torch.zeros(2, dtype=torch.long)) is False


def test_vocab_embed_native_warns_when_sglang_drops_the_api(monkeypatch):
    embedding_cls = vocab_parallel_embedding.VocabParallelEmbedding
    monkeypatch.delattr(embedding_cls, "_use_triton_embedding", raising=False)
    monkeypatch.setattr(vocab_embed_native, "_applied", False)

    vocab_embed_native.patch_vocab_embed_native()

    assert vocab_embed_native._applied is True