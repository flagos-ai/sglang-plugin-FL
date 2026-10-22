# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real Hygon FlashAttention MTP numerics against an independent dense oracle.

These are eager component tests, not model-level or graph acceptance. They use
production speculative metadata and forward methods, real vendor kernels and
nonconsecutive physical KV pages. No SGLang worker or model is started here.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.fixture
def hcu_module():
    if not torch.cuda.is_available():
        pytest.skip("Hygon accelerator unavailable")
    from sglang_fl.utils import get_device_info

    if get_device_info().vendor_name != "hygon":
        pytest.skip("Hygon vendor kernel test")
    from sglang_fl.dispatch.backends.vendor.hygon.impl import attention_backend

    return attention_backend


class KVPool:
    def __init__(self, capacity, head_dim, device):
        self.k = torch.zeros(capacity, 1, head_dim, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        self.writes = 0

    def get_kv_buffer(self, layer_id):
        assert layer_id == 0
        return self.k, self.v

    def set_kv_buffer(self, layer, locations, k, v):
        self.k[locations] = k
        self.v[locations] = v
        self.writes += 1


def make_backend(module, head_dim, device):
    # The full runner requires distributed/model globals. This component test
    # supplies its ownership/config fields, leaving production metadata/forward
    # methods and all vendor arithmetic intact.
    backend = module.HCUAttnBackend.__new__(module.HCUAttnBackend)
    backend.device = device
    backend.max_context_len = 256
    backend.num_head = 2
    backend.use_mla = False
    backend.dcp_size = 1
    backend.sliding_window_size = None
    backend.enable_deterministic = False
    backend._translate_kv_loc = None
    backend._flash_attn_input_dtype = torch.bfloat16
    backend._flash_attn_kv_cache_dtype = torch.bfloat16
    backend._flash_attn_head_dim = head_dim
    backend._flash_attn_v_head_dim = head_dim
    backend._flash_attn_page_size = 64
    backend._flash_attn_speculative_topk = 1
    backend._flash_attn_speculative_enabled = True
    backend._flash_attn_speculative_phase = None
    backend._flash_attn_eager_page_table = None
    backend._flash_attn_forward_page_table = None
    backend._flash_attn_forward_seq_lens = None
    backend._flash_attn_prefill_metadata_ready = False
    backend._flash_attn_prefill_uses_paged_kv = False
    backend._flash_attn_prefill_logged = False
    backend._flash_attn_prefill_paged_logged = False
    backend._flash_attn_decode_logged = False
    backend._flash_attn_debug_page_table = False
    backend._flash_attn_debug_require_disjoint_pages = True
    backend.token_to_kv_pool = KVPool(9 * 64, head_dim, device)
    # Every logical page is contiguous, while consecutive logical pages are
    # deliberately far apart physically. Row order differs from page order.
    rows = []
    for page_ids in ((1, 5, 2, 6), (3, 7, 4, 8)):
        rows.append(
            torch.cat(
                [torch.arange(p * 64, (p + 1) * 64, device=device) for p in page_ids]
            )
        )
    backend.req_to_token = torch.stack(rows)
    backend._flash_attn_varlen_func = module._load_flash_attn_varlen_func()
    backend._flash_attn_with_kvcache = module._load_flash_attn_with_kvcache()
    return backend


def make_layer(head_dim):
    return SimpleNamespace(
        layer_id=0,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        k_scale=None,
        v_scale=None,
        scaling=head_dim**-0.5,
        logit_cap=0.0,
    )


def make_batch(module, backend, prefixes, width):
    device = backend.device
    prefixes = list(prefixes)
    return SimpleNamespace(
        batch_size=len(prefixes),
        forward_mode=module.ForwardMode.TARGET_VERIFY,
        global_forward_mode=None,
        spec_info=SimpleNamespace(draft_token_num=width, custom_mask=None),
        seq_lens=torch.tensor(prefixes, device=device, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(prefixes, dtype=torch.int64),
        seq_lens_sum=sum(prefixes),
        req_pool_indices=torch.arange(len(prefixes), device=device),
        out_cache_loc=torch.cat(
            [backend.req_to_token[i, n : n + width] for i, n in enumerate(prefixes)]
        ),
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=None,
        extend_seq_lens=None,
        encoder_lens=None,
    )


def dense_reference(q, prefix_k, prefix_v, new_k, new_v, scale):
    """FP32 oracle with explicit prefix + lower triangular chain masking."""
    keys = torch.cat([prefix_k, new_k]).float().repeat_interleave(2, dim=1)
    values = torch.cat([prefix_v, new_v]).float().repeat_interleave(2, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), keys) * scale
    query = torch.arange(q.shape[0], device=q.device)[:, None]
    key = torch.arange(keys.shape[0], device=q.device)[None, :]
    allowed = key <= len(prefix_k) + query
    scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(dim=-1), values)


def initialize_prefixes(backend, prefixes, head_dim):
    prefix_k, prefix_v = [], []
    for row, length in enumerate(prefixes):
        k = torch.randn(
            length, 1, head_dim, device=backend.device, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        backend.token_to_kv_pool.k[backend.req_to_token[row, :length]] = k
        backend.token_to_kv_pool.v[backend.req_to_token[row, :length]] = v
        prefix_k.append(k)
        prefix_v.append(v)
    return prefix_k, prefix_v


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("prefixes", [(63, 65), (64, 129)])
def test_target_verify_chain_matches_dense_reference(hcu_module, head_dim, prefixes):
    torch.manual_seed(812)
    backend = make_backend(hcu_module, head_dim, torch.device("cuda", 0))
    width = 4
    old_k, old_v = initialize_prefixes(backend, prefixes, head_dim)
    batch = make_batch(hcu_module, backend, prefixes, width)
    q = torch.randn(8, 2, head_dim, device=backend.device, dtype=torch.bfloat16)
    k = torch.randn(8, 1, head_dim, device=backend.device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    actual = backend.forward_extend(q.flatten(1), k, v, make_layer(head_dim), batch)
    expected = torch.cat(
        [
            dense_reference(
                q[i * width : (i + 1) * width],
                old_k[i],
                old_v[i],
                k[i * width : (i + 1) * width],
                v[i * width : (i + 1) * width],
                head_dim**-0.5,
            )
            for i in range(len(prefixes))
        ]
    ).flatten(1)
    # Fixed BF16 tolerance; failures must not silently relax the oracle.
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(backend.token_to_kv_pool.k[batch.out_cache_loc], k)
    assert backend.token_to_kv_pool.writes == 1
    print(
        f"HCU_MTP_NUMERICAL target_verify head_dim={head_dim} prefixes={prefixes} width=4 max_abs_error={(actual.float() - expected).abs().max().item():.8f}"
    )


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_rejected_kv_slots_do_not_contaminate_next_decode(hcu_module, accepted):
    torch.manual_seed(445)
    head_dim, width = 64, 4
    prefixes = (63, 65)
    backend = make_backend(hcu_module, head_dim, torch.device("cuda", 0))
    old_k, old_v = initialize_prefixes(backend, prefixes, head_dim)
    batch = make_batch(hcu_module, backend, prefixes, width)
    q = torch.randn(8, 2, head_dim, device=backend.device, dtype=torch.bfloat16)
    proposed_k = torch.randn(
        8, 1, head_dim, device=backend.device, dtype=torch.bfloat16
    )
    proposed_v = torch.randn_like(proposed_k)
    backend._prepare_flash_attn_speculative_metadata(batch)
    backend.forward_extend(
        q.flatten(1), proposed_k, proposed_v, make_layer(head_dim), batch
    )

    # A real allocator can reuse rejected slots. Poison all rejected entries so
    # reading the old verify length instead of accepted+1 is visible numerically.
    for row, prefix in enumerate(prefixes):
        rejected = backend.req_to_token[row, prefix + accepted : prefix + width]
        backend.token_to_kv_pool.k[rejected] = 1000
        backend.token_to_kv_pool.v[rejected] = -1000
    full_lengths = [prefix + accepted + 1 for prefix in prefixes]
    next_batch = make_batch(hcu_module, backend, prefixes, 1)
    next_batch.forward_mode = hcu_module.ForwardMode.DECODE
    next_batch.spec_info = SimpleNamespace(
        kv_indptr=torch.tensor(
            [0, full_lengths[0], sum(full_lengths)],
            device=backend.device,
            dtype=torch.int32,
        ),
        kv_indices=torch.cat(
            [backend.req_to_token[i, :n] for i, n in enumerate(full_lengths)]
        ),
    )
    next_batch.out_cache_loc = torch.stack(
        [backend.req_to_token[i, n - 1] for i, n in enumerate(full_lengths)]
    )
    next_q = torch.randn(2, 2, head_dim, device=backend.device, dtype=torch.bfloat16)
    next_k = torch.randn(2, 1, head_dim, device=backend.device, dtype=torch.bfloat16)
    next_v = torch.randn_like(next_k)
    backend._prepare_flash_attn_speculative_metadata(next_batch)
    actual = backend.forward_decode(
        next_q.flatten(1), next_k, next_v, make_layer(head_dim), next_batch
    )
    expected = []
    for i in range(2):
        kept_k = torch.cat([old_k[i], proposed_k[i * width : i * width + accepted]])
        kept_v = torch.cat([old_v[i], proposed_v[i * width : i * width + accepted]])
        expected.append(
            dense_reference(
                next_q[i : i + 1],
                kept_k,
                kept_v,
                next_k[i : i + 1],
                next_v[i : i + 1],
                head_dim**-0.5,
            )
        )
    expected = torch.cat(expected).flatten(1)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
    assert backend.token_to_kv_pool.writes == 2
    print(
        f"HCU_MTP_NUMERICAL draft_decode accepted={accepted} max_abs_error={(actual.float() - expected).abs().max().item():.8f}"
    )


@pytest.mark.parametrize("prefixes", [(0, 0), (0, 64)])
def test_initial_draft_prefill_chunk_matches_dense_reference(hcu_module, prefixes):
    torch.manual_seed(952)
    head_dim = 64
    widths = [7, 5]
    backend = make_backend(hcu_module, head_dim, torch.device("cuda", 0))
    old_k, old_v = initialize_prefixes(backend, prefixes, head_dim)
    batch = make_batch(hcu_module, backend, prefixes, 1)
    batch.forward_mode = hcu_module.ForwardMode.EXTEND
    batch.spec_info = SimpleNamespace(num_tokens_per_req=1)
    batch.extend_prefix_lens_cpu = list(prefixes)
    batch.extend_prefix_lens = torch.tensor(prefixes, device=backend.device)
    batch.extend_seq_lens_cpu = widths
    batch.extend_seq_lens = torch.tensor(widths, device=backend.device)
    batch.extend_num_tokens = sum(widths)
    batch.extend_start_loc = torch.tensor([0, widths[0]], device=backend.device)
    full_lengths = [p + n for p, n in zip(prefixes, widths)]
    batch.seq_lens = torch.tensor(full_lengths, device=backend.device)
    batch.seq_lens_cpu = torch.tensor(full_lengths)
    batch.out_cache_loc = torch.cat(
        [
            backend.req_to_token[i, p : p + n]
            for i, (p, n) in enumerate(zip(prefixes, widths))
        ]
    )
    q = torch.randn(
        sum(widths), 2, head_dim, device=backend.device, dtype=torch.bfloat16
    )
    k = torch.randn(
        sum(widths), 1, head_dim, device=backend.device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    assert backend._prepare_flash_attn_speculative_metadata(batch)
    actual = backend.forward_extend(q.flatten(1), k, v, make_layer(head_dim), batch)
    expected, offset = [], 0
    for i, n in enumerate(widths):
        expected.append(
            dense_reference(
                q[offset : offset + n],
                old_k[i],
                old_v[i],
                k[offset : offset + n],
                v[offset : offset + n],
                head_dim**-0.5,
            )
        )
        offset += n
    expected = torch.cat(expected).flatten(1)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
    assert backend._flash_attn_prefill_cu_seqlens_q.tolist() == [0, 7, 12]
    assert backend.token_to_kv_pool.writes == 1
    print(
        f"HCU_MTP_NUMERICAL initial_draft_prefill prefixes={prefixes} widths={widths} max_abs_error={(actual.float() - expected).abs().max().item():.8f}"
    )
