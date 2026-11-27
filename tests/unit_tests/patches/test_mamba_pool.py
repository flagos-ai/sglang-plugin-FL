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

# Chunked MambaPool.copy_from (issue #209, backport of sgl-project/sglang#47984).

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang_fl.patches import mamba_pool as mamba_pool_patch  # noqa: E402
from sglang_fl.patches.mamba_pool import chunked_copy_from  # noqa: E402

# Small pool: 3 layers x 12 slots x (4, 5, 6) fp32 -> one slot = 1440 B.
LAYERS, SLOTS, STATE = 3, 12, (4, 5, 6)
SLOT_BYTES = LAYERS * 4 * 5 * 6 * 4


def _make_state(device="cpu", transposed=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    temporal = torch.randn((LAYERS, SLOTS) + STATE, generator=g).to(device)
    if transposed:
        # Ascend speculative layout: a non-contiguous transpose(-1, -2) view.
        temporal = temporal.transpose(-1, -2)
    conv = [torch.randn((LAYERS, SLOTS, 7, 3), generator=g).to(device)]
    return temporal, conv


class _FakePool:
    """Mirrors the non-fused body of SGLang 0.5.18 MambaPool.copy_from."""

    def __init__(self, temporal, conv, replayssm=False, debug=False):
        self.mamba_cache = SimpleNamespace(temporal=temporal, conv=conv)
        device = temporal.device
        self.replayssm_write_pos = (
            torch.ones(SLOTS, dtype=torch.int32, device=device) if replayssm else None
        )
        self.debug_memory_pool = debug
        self.calls = []

    def copy_from(self, src_indices, dst_indices):
        self.calls.append(dst_indices.numel())
        for c in self.mamba_cache.conv:
            c[:, dst_indices] = c[:, src_indices]
        t = self.mamba_cache.temporal
        t[:, dst_indices] = t[:, src_indices]
        if self.replayssm_write_pos is not None:
            self.replayssm_write_pos[dst_indices] = 0


def _original(self, src, dst):
    return _FakePool.copy_from(self, src, dst)


def _pools(**kw):
    """Two identical pools: reference (one-shot) and patched (chunked)."""
    transposed = kw.pop("transposed", False)
    ref = _FakePool(*_make_state(transposed=transposed), **kw)
    new = _FakePool(*_make_state(transposed=transposed), **kw)
    return ref, new


@pytest.fixture
def chunk_slots(monkeypatch):
    def _set(n_slots_per_chunk):
        monkeypatch.setattr(
            mamba_pool_patch, "_CHUNK_BYTES", n_slots_per_chunk * SLOT_BYTES
        )

    return _set


class TestChunkedCopyFromCorrectness:
    @pytest.mark.parametrize("transposed", [False, True])
    @pytest.mark.parametrize("step", [1, 2, 3])
    def test_matches_one_shot_gather(self, chunk_slots, transposed, step):
        chunk_slots(step)
        ref, new = _pools(transposed=transposed, replayssm=True)
        # Duplicate src slots are legal (several reqs hit the same prefix).
        src = torch.tensor([8, 9, 9, 10, 11, 8, 10])
        dst = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        ref.copy_from(src, dst)
        chunked_copy_from(_original, new, src, dst)

        assert torch.equal(ref.mamba_cache.temporal, new.mamba_cache.temporal)
        assert torch.equal(ref.mamba_cache.conv[0], new.mamba_cache.conv[0])
        assert torch.equal(ref.replayssm_write_pos, new.replayssm_write_pos)
        expected = [min(step, 7 - s) for s in range(0, 7, step)]
        assert new.calls == expected

    def test_untouched_slots_stay_unchanged(self, chunk_slots):
        chunk_slots(1)
        _, new = _pools()
        before = new.mamba_cache.temporal.clone()
        chunked_copy_from(_original, new, torch.tensor([10, 11]), torch.tensor([2, 5]))
        untouched = [i for i in range(SLOTS) if i not in (2, 5)]
        assert torch.equal(
            new.mamba_cache.temporal[:, untouched], before[:, untouched]
        )


class TestChunkedCopyFromPassthrough:
    def test_small_batch_single_call(self, chunk_slots):
        chunk_slots(4)
        _, new = _pools()
        chunked_copy_from(_original, new, torch.tensor([9, 10]), torch.tensor([0, 1]))
        assert new.calls == [2]

    def test_upstream_fixed_base_is_left_alone(self, chunk_slots):
        chunk_slots(1)
        _, new = _pools()
        new._copy_temporal_slots = lambda **kw: None
        chunked_copy_from(
            _original, new, torch.tensor([8, 9, 10]), torch.tensor([0, 1, 2])
        )
        assert new.calls == [3]

    def test_empty_temporal_passthrough(self, chunk_slots):
        chunk_slots(1)
        pool = _FakePool(torch.zeros((0, SLOTS) + STATE), [])
        chunked_copy_from(_original, pool, torch.tensor([8, 9]), torch.tensor([0, 1]))
        assert pool.calls == [2]


def test_debug_mode_rejects_overlapping_slots(chunk_slots):
    chunk_slots(1)
    _, new = _pools(debug=True)
    with pytest.raises(AssertionError, match="disjoint"):
        chunked_copy_from(_original, new, torch.tensor([1, 2]), torch.tensor([2, 3]))


class TestAgainstSGLangMambaPool:
    """Run the real SGLang MambaPool.copy_from as ``original``."""

    def test_real_copy_from_matches_one_shot(self, chunk_slots):
        memory_pool = pytest.importorskip("sglang.srt.mem_cache.memory_pool")
        MambaPool = memory_pool.MambaPool
        if hasattr(MambaPool, "_copy_temporal_slots"):
            pytest.skip("base SGLang already carries #47984")
        chunk_slots(2)

        def _self(temporal, conv):
            return SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=temporal, conv=conv),
                replayssm_write_pos=None,
                replayssm_cache_base=None,
                replayssm_is_flush=None,
                debug_memory_pool=False,
                _should_fuse_slot_ops=lambda: False,
            )

        ref, new = _self(*_make_state()), _self(*_make_state())
        src = torch.tensor([7, 8, 9, 10, 11])
        dst = torch.tensor([0, 1, 2, 3, 4])
        MambaPool.copy_from(ref, src, dst)
        chunked_copy_from(MambaPool.copy_from, new, src, dst)
        assert torch.equal(ref.mamba_cache.temporal, new.mamba_cache.temporal)
        assert torch.equal(ref.mamba_cache.conv[0], new.mamba_cache.conv[0])

    def test_hook_registered_on_mamba_pool(self, monkeypatch):
        hook_registry = pytest.importorskip("sglang.srt.plugins.hook_registry")
        registered = []
        monkeypatch.setattr(
            hook_registry.HookRegistry,
            "register",
            classmethod(lambda cls, *a, **kw: registered.append(a)),
        )
        mamba_pool_patch.patch_mamba_pool_copy_from()
        assert registered == [
            (
                "sglang.srt.mem_cache.memory_pool.MambaPool.copy_from",
                chunked_copy_from,
                hook_registry.HookType.AROUND,
            )
        ]


def _accelerator():
    if torch.cuda.is_available():
        return "cuda", torch.cuda
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu", npu
    pytest.skip("needs a CUDA or NPU device")


@pytest.mark.gpu
@pytest.mark.parametrize("transposed", [False, True])
def test_peak_memory_is_bounded_by_chunk(monkeypatch, transposed):
    """One-shot peak grows with n_slots; chunked peak stays at one chunk."""
    device, acc = _accelerator()
    big_slots, n = 40, 16
    temporal = torch.zeros((LAYERS, big_slots, 64, 128, 128), device=device)
    if transposed:
        temporal = temporal.transpose(-1, -2)
    slot_bytes = temporal.element_size() * (temporal.numel() // big_slots)
    pool = _FakePool.__new__(_FakePool)
    pool.mamba_cache = SimpleNamespace(temporal=temporal, conv=[])
    pool.replayssm_write_pos = None
    pool.debug_memory_pool = False
    pool.calls = []
    src = torch.arange(big_slots - n, big_slots, device=device)
    dst = torch.arange(0, n, device=device)

    def _peak(fn):
        acc.synchronize()
        base = acc.memory_allocated()
        acc.reset_peak_memory_stats()
        fn()
        acc.synchronize()
        return acc.max_memory_allocated() - base

    monkeypatch.setattr(mamba_pool_patch, "_CHUNK_BYTES", slot_bytes)
    one_shot = _peak(lambda: _FakePool.copy_from(pool, src, dst))
    chunked = _peak(lambda: chunked_copy_from(_original, pool, src, dst))

    assert one_shot >= n * slot_bytes
    assert chunked <= 2 * slot_bytes
