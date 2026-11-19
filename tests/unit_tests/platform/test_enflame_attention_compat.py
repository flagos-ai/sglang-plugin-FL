"""CPU contracts for ordinary GCU decode metadata; no device graph is executed."""

import ast
from pathlib import Path
from typing import Optional

import pytest
import torch


@pytest.fixture
def set_metadata():
    # Isolate the real function from unavailable vendor Attention dependencies.
    root = Path(__file__).resolve().parents[3]
    path = root / "sglang_fl/dispatch/backends/vendor/enflame/patches/flashattention_backend.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "normal_decode_set_metadata"
    )
    namespace = dict(torch=torch, Optional=Optional, SWAKVPool=object)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("mapped", [False, True])
def test_metadata_page_mapping(set_metadata, mapped):
    cache = torch.zeros(2, dtype=torch.int32)
    cumulative = torch.zeros(3, dtype=torch.int32)
    pages = torch.full((2, 3), -99, dtype=torch.int32)
    req_to_token = torch.tensor([[0, 1, 4, 5], [8, 9, 12, 13]])
    mapping = torch.arange(7, 0, -1) if mapped else None
    set_metadata(
        cache, cumulative, pages, req_to_token, torch.tensor([1, 0]),
        torch.tensor([0, 2]), 2, torch.tensor([3, 2]), 1, 2,
        v2p_page_table=mapping, kernel_page_multiplier=2 if mapped else 1,
    )
    assert cache.tolist() == [4, 3]
    assert cumulative.tolist() == [0, 4, 7]
    assert pages[:, :2].tolist() == ([[6, 2], [14, 10]] if mapped else [[4, 6], [0, 2]])
    assert pages[:, 2].tolist() == [-99, -99]


@pytest.mark.parametrize("mapped", [False, True])
def test_repeated_metadata_updates_refresh_existing_buffers(set_metadata, mapped):
    """Model capture/replay preparation using the same preallocated CPU buffers."""
    cache = torch.full((2,), -99, dtype=torch.int32)
    cumulative = torch.tensor([0, -99, -99], dtype=torch.int32)
    pages = torch.full((2, 4), -99, dtype=torch.int32)
    addresses = tuple(t.data_ptr() for t in (cache, cumulative, pages))
    # Keep references as a graph would; later writes must reach this storage.
    held_cache, held_cumulative, held_pages = cache.view_as(cache), cumulative.view_as(cumulative), pages.view_as(pages)
    req_to_token = torch.tensor([
        [0, 1, 4, 5, 8, 9],
        [12, 13, 16, 17, 20, 21],
        [24, 25, 28, 29, 32, 33],
    ])
    mapping = torch.arange(32) * 3 + 1 if mapped else None
    options = dict(v2p_page_table=mapping, kernel_page_multiplier=2 if mapped else 1)
    stride = torch.tensor([0, 2, 4])

    set_metadata(
        cache, cumulative, pages, req_to_token, torch.tensor([2, 0]),
        stride, 3, torch.tensor([5, 3]), 1, 2, **options,
    )
    expected_capture = [[74, 86, 98], [2, 14, 26]] if mapped else [[12, 14, 16], [0, 2, 4]]
    assert held_cache.tolist() == [6, 4]
    assert held_cumulative.tolist() == [0, 6, 10]
    assert held_pages[:, :3].tolist() == expected_capture
    assert tuple(t.data_ptr() for t in (cache, cumulative, pages)) == addresses

    # Replay preparation must use both the new request order and updated KV locations.
    req_to_token[1, 0] = 40
    req_to_token[2, 2] = 44
    seq_lens = torch.tensor([2, 5])
    input_before = req_to_token.clone(), seq_lens.clone()
    set_metadata(
        cache, cumulative, pages, req_to_token, torch.tensor([1, 2]),
        stride, 3, seq_lens, 0, 2, **options,
    )
    expected_replay = [[122, 50, 62], [74, 134, 98]] if mapped else [[20, 8, 10], [12, 22, 16]]
    assert held_cache.tolist() == [2, 5]
    assert held_cumulative.tolist() == [0, 2, 7]
    assert held_pages[:, :3].tolist() == expected_replay
    assert held_pages[:, 3].tolist() == [-99, -99]
    assert tuple(t.data_ptr() for t in (cache, cumulative, pages)) == addresses
    torch.testing.assert_close(req_to_token, input_before[0], rtol=0, atol=0)
    torch.testing.assert_close(seq_lens, input_before[1], rtol=0, atol=0)


def test_smaller_then_larger_batch_updates_only_active_views(set_metadata):
    """Views model active extents, not a promise of real graph batch resizing."""
    cache = torch.zeros(2, dtype=torch.int32)
    cumulative = torch.zeros(3, dtype=torch.int32)
    pages = torch.full((2, 4), -99, dtype=torch.int32)
    addresses = tuple(t.data_ptr() for t in (cache, cumulative, pages))
    table = torch.tensor([[0, 1, 4, 5, 8, 9], [12, 13, 16, 17, 20, 21]])
    stride = torch.tensor([0, 2, 4])
    set_metadata(cache, cumulative, pages, table, torch.tensor([0, 1]), stride, 3, torch.tensor([5, 3]), 0, 2)
    before = cache.clone(), cumulative.clone(), pages.clone()

    set_metadata(cache[:1], cumulative[:2], pages[:1], table, torch.tensor([1]), stride, 1, torch.tensor([1]), 1, 2)
    assert cache[0].item() == 2
    assert cumulative[:2].tolist() == [0, 2]
    assert pages[0, 0].item() == 6
    # The implementation leaves inactive storage alone; consumers must use active lengths.
    torch.testing.assert_close(cache[1:], before[0][1:], rtol=0, atol=0)
    torch.testing.assert_close(cumulative[2:], before[1][2:], rtol=0, atol=0)
    torch.testing.assert_close(pages[0, 1:], before[2][0, 1:], rtol=0, atol=0)
    torch.testing.assert_close(pages[1], before[2][1], rtol=0, atol=0)

    set_metadata(cache, cumulative, pages, table, torch.tensor([1, 0]), stride, 2, torch.tensor([3, 2]), 1, 2)
    assert cache.tolist() == [4, 3]
    assert cumulative.tolist() == [0, 4, 7]
    assert pages[:, :2].tolist() == [[6, 8], [0, 2]]
    assert pages[:, 3].tolist() == [-99, -99]
    assert tuple(t.data_ptr() for t in (cache, cumulative, pages)) == addresses


def _attention_function(name, **namespace):
    path = Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/backends/vendor/enflame/patches/flashattention_backend.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def attention_init(monkeypatch):
    import sys
    from types import ModuleType

    modules = {name: ModuleType(name) for name in (
        "sgl_kernel", "sgl_kernel.flash_attn", "flash_attn", "flash_attn.vllm_flash_attn"
    )}
    modules["sgl_kernel"].__path__ = []
    modules["flash_attn"].__path__ = []
    modules["sgl_kernel"].flash_attn = modules["sgl_kernel.flash_attn"]
    metadata, varlen, kvcache = object(), object(), object()
    modules["flash_attn.vllm_flash_attn"].get_scheduler_metadata = metadata
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    init = _attention_function("__init__", flash_attn_varlen_func=varlen,
                               flash_attn_with_kvcache=kvcache)
    return init, modules["sgl_kernel.flash_attn"], metadata, varlen, kvcache


@pytest.mark.parametrize("version", [3, 4])
def test_attention_wraps_upstream_init_and_only_replaces_fa3(attention_init, version):
    from types import SimpleNamespace

    init, kernel_module, metadata, varlen, kvcache = attention_init
    runner, positional, option, pool, retained = (object() for _ in range(5))
    original_varlen, original_kvcache, original_metadata = (object() for _ in range(3))
    backend, calls = SimpleNamespace(), []

    def original(self, actual_runner, *args, **kwargs):
        assert self is backend and actual_runner is runner
        assert args == (positional,) and kwargs == {"new_option": option}
        # The helper must be installed before upstream initialization uses it.
        assert kernel_module.get_scheduler_metadata is metadata
        calls.append(self)
        self.fa_impl_ver = version
        self.token_to_kv_pool = pool
        self.upstream_retained = retained
        self.flash_attn_varlen_func = original_varlen
        self.flash_attn_with_kvcache = original_kvcache
        self._get_scheduler_metadata = original_metadata

    init(original, backend, runner, positional, new_option=option)
    assert calls == [backend]
    assert backend.token_to_kv_pool is pool and backend.upstream_retained is retained
    assert backend.flash_attn_varlen_func is (varlen if version == 3 else original_varlen)
    assert backend.flash_attn_with_kvcache is (kvcache if version == 3 else original_kvcache)
    assert backend._get_scheduler_metadata is (metadata if version == 3 else original_metadata)


def test_attention_init_propagates_upstream_failure(attention_init):
    from types import SimpleNamespace

    init, _, _, _, _ = attention_init
    backend, error = SimpleNamespace(), RuntimeError("upstream init failed")

    def original(*args, **kwargs):
        raise error

    with pytest.raises(RuntimeError) as caught:
        init(original, backend, object())
    assert caught.value is error
    assert not hasattr(backend, "flash_attn_varlen_func")


def test_attention_initialization_hook_uses_around_semantics(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    calls = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookRegistry = SimpleNamespace(register=lambda *args: calls.append(args))
    registry.HookType = SimpleNamespace(AROUND=object(), REPLACE=object())
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    callbacks = {name: object() for name in (
        "__init__", "forward_extend", "forward_decode", "_compute_scheduler_metadata",
        "_create_flashattention_v3_backend_hook", "normal_decode_set_metadata"
    )}
    _attention_function("patch_flashattention_backend", **callbacks)()
    matches = [entry for entry in calls if entry[0].endswith("FlashAttentionBackend.__init__")]
    assert matches == [("sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.__init__",
                        callbacks["__init__"], registry.HookType.AROUND)]


@pytest.mark.parametrize("stage", ["extend", "decode"])
@pytest.mark.parametrize("save_kv_cache", [False, True])
def test_attention_reads_backend_pool_without_batch_pool(stage, save_kv_cache):
    from types import SimpleNamespace

    q, k, v = (torch.arange(4, dtype=torch.float32).reshape(2, 1, 2) + offset
               for offset in (0, 10, 20))
    key_cache = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    value_cache = key_cache + 100
    cache_loc = object()
    writes, reads, kernel_calls = [], [], []
    expected_output = q + 0.5

    def write(layer, loc, actual_k, actual_v, k_scale, v_scale):
        assert loc is cache_loc and actual_k is k and actual_v is v
        writes.append(layer)

    def read(layer_id):
        reads.append(layer_id)
        return key_cache, value_cache

    def attention(**kwargs):
        kernel_calls.append(kwargs)
        assert kwargs["k"].data_ptr() == key_cache.data_ptr()
        assert kwargs["v"].data_ptr() == value_cache.data_ptr()
        return expected_output

    metadata = SimpleNamespace(
        page_table=torch.tensor([[0, 1], [2, 3]]),
        cu_seqlens_q=torch.tensor([0, 1, 2]),
        cache_seqlens_int32=torch.tensor([2, 2]), cu_seqlens_k=torch.tensor([0, 2, 4]),
        max_seq_len_q=1, scheduler_metadata=None,
    )
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(set_kv_buffer=write, get_kv_buffer=read),
        forward_metadata=metadata, use_mla=False, attn_cp_size=1, fa_skip_kv_cache=False,
        kv_cache_dtype_str="auto", fa_impl_ver=3, has_local_attention=False,
        topk=0, page_size=2, num_splits=1, max_context_len=8,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False,
                                     is_target_verify=lambda: False),
        out_cache_loc=cache_loc, spec_info=None,
    )
    assert not hasattr(batch, "token_to_kv_pool")
    layer = SimpleNamespace(layer_id=7, is_cross_attention=False, sliding_window_size=-1,
                            k_scale=None, v_scale=None, head_dim=2, v_head_dim=2,
                            tp_q_head_num=1, tp_k_head_num=1, tp_v_head_num=1,
                            attn_type="decoder", scaling=0.5, logit_cap=0)
    fn = _attention_function("forward_" + stage, torch=torch,
                             AttentionType=SimpleNamespace(ENCODER_ONLY="encoder"),
                             flash_attn_varlen_func=attention)
    actual = fn(backend, q, k, v, layer, batch, save_kv_cache=save_kv_cache)
    assert writes == ([layer] if save_kv_cache else [])
    assert reads == [7] and len(kernel_calls) == 1
    assert kernel_calls[0]["block_table"] is metadata.page_table
    torch.testing.assert_close(actual, expected_output.reshape(2, 2))
