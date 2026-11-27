# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The HCU ``sgl_kernel`` fallbacks against independent references, on device.

Each fallback replaces an operator SGLang imports from ``sgl_kernel``, which the
Hygon Runtime does not ship. These tests call the fallback through the
placeholder and check it against plain PyTorch, or against SGLang's own
reference data for the tree kernels. They are component tests: no model or
worker is started.
"""

import pytest
import torch

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.fixture(scope="module")
def kernel():
    if not torch.cuda.is_available():
        pytest.skip("Hygon accelerator unavailable")
    from sglang_fl.utils import get_device_info

    if get_device_info().vendor_name != "hygon":
        pytest.skip("Hygon fallback test")
    from sglang_fl.dispatch.backends.vendor.hygon import import_guard

    if "sgl_kernel" not in import_guard.install_import_guards():
        pytest.skip("a real sgl_kernel is installed")
    import sgl_kernel

    return sgl_kernel


def test_fast_topk_matches_torch_topk(kernel):
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(8, 16, generator=generator).to("cuda")
    for topk in (1, 2, 4):
        got = kernel.fast_topk(values, topk, dim=-1)
        want = torch.topk(values, topk, dim=-1)
        assert torch.equal(got.values, want.values)
        assert torch.equal(got.indices, want.indices)


@pytest.mark.parametrize("tokens", [1, 7, 300])
def test_moe_align_block_size_groups_tokens_by_expert(kernel, tokens):
    """Every block holds tokens of one expert and no token is lost."""
    num_experts, topk, block_size = 8, 4, 64
    generator = torch.Generator().manual_seed(tokens)
    topk_ids = torch.randint(
        0, num_experts, (tokens, topk), generator=generator, dtype=torch.int64
    ).to("cuda")
    max_blocks = topk * tokens + num_experts * (block_size - 1)
    sorted_token_ids = torch.zeros(
        max_blocks * block_size, dtype=torch.int32, device="cuda"
    )
    experts_ids = torch.zeros(max_blocks, dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")

    kernel.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
    )
    torch.cuda.synchronize()

    padded = int(num_tokens_post_pad.item())
    assert padded % block_size == 0 and padded > 0
    pad_value = topk_ids.numel()
    valid = sorted_token_ids[:padded].long()
    real = valid[valid < pad_value]
    # No token is lost: each appears once per expert it was routed to.
    assert real.numel() == tokens * topk
    assert torch.equal(
        torch.bincount(real // topk, minlength=tokens),
        torch.full((tokens,), topk, dtype=torch.long, device="cuda"),
    )
    per_block = valid.view(-1, block_size)
    for block in range(padded // block_size):
        rows = per_block[block]
        rows = rows[rows < pad_value]
        assert (topk_ids.flatten()[rows.long()] == experts_ids[block]).all()


def test_silu_and_mul_and_moe_sum_reduce_match_torch(kernel):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(257, 1024, generator=generator).to("cuda", torch.bfloat16)
    out = torch.empty(257, 512, device="cuda", dtype=torch.bfloat16)
    kernel.silu_and_mul(x, out)
    want = torch.nn.functional.silu(x[:, :512].float()) * x[:, 512:].float()
    assert torch.allclose(out.float(), want, atol=2e-2, rtol=2e-2)

    weights = torch.randn(64, 8, 512, generator=generator).to("cuda", torch.bfloat16)
    out = torch.empty(64, 512, device="cuda", dtype=torch.bfloat16)
    kernel.moe_sum_reduce(weights, out, 0.5)
    assert torch.allclose(
        out.float(), weights.float().sum(1) * 0.5, atol=2e-1, rtol=2e-2
    )


# Candidate layout of eagle_utils.organize_draft_results: candidates 0..N-1,
# node position = candidate + 1 (position 0 is the bonus token).
def _build_tree(kernel, topk, parents):
    from sglang.srt.speculative import eagle_utils

    steps, batch, prompt = 3, 2, 7
    num_candidates = steps * topk
    bonus = torch.arange(100, 100 + batch, device="cuda")
    seq_lens = torch.full((batch,), prompt, device="cuda", dtype=torch.long)
    tree = eagle_utils.build_tree_kernel_efficient(
        bonus,
        torch.tensor([parents] * batch, device="cuda"),
        torch.tensor([list(range(num_candidates))] * batch, device="cuda"),
        torch.arange(num_candidates, device="cuda")
        .add(1000)
        .expand(batch, -1)
        .contiguous(),
        seq_lens,
        batch * prompt,
        topk,
        steps,
        1 + num_candidates,
    )
    torch.cuda.synchronize()
    return tree


def test_build_tree_links_chain_and_branches(kernel):
    # The parent of the node for candidate i is parents[i // topk]; candidate
    # indices 0..topk-1 hang off the root.
    tree = _build_tree(kernel, 1, [-1, 0, 1])
    assert tree[3][0].tolist() == [1, 2, 3, -1]  # retrieve_next_token
    assert tree[4][0].tolist() == [-1, -1, -1, -1]  # retrieve_next_sibling
    assert tree[1][:4].tolist() == [7, 8, 9, 10]  # positions

    tree = _build_tree(kernel, 2, [-1, 0, 2])
    assert tree[3][0].tolist() == [1, 3, -1, 5, -1, -1, -1]
    assert tree[4][0].tolist() == [-1, 2, -1, 4, -1, 6, -1]
    assert tree[1][:7].tolist() == [7, 8, 8, 9, 9, 10, 10]


# SGLang's reference case for the kernel, from
# sglang/kernels/aot/tests/speculative/test_eagle_utils.py.
def test_verify_tree_greedy_matches_sglang_reference(kernel):
    candidates = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], device="cuda"
    )
    tree = (
        torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], device="cuda"),
        torch.tensor([[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]], device="cuda"),
        torch.tensor([[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]], device="cuda"),
    )
    target = torch.tensor(
        [[3, 18, 18, 4, 5, 18], [11, 18, 18, 18, 12, 18]], device="cuda"
    )
    predicts = torch.full((12,), -1, dtype=torch.int64, device="cuda")
    accept_index = torch.full((2, 4), -1, dtype=torch.int64, device="cuda")
    accept_token_num = torch.zeros(2, dtype=torch.int64, device="cuda")

    kernel.verify_tree_greedy(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index=tree[0],
        retrive_next_token=tree[1],
        retrive_next_sibling=tree[2],
        target_predict=target,
    )
    torch.cuda.synchronize()
    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [[0, 3, 4, 5], [6, 10, 11, -1]]
    assert accept_token_num.tolist() == [3, 2]


# Tree of _build_tree(kernel, 2, [-1, 0, 2]): the root's children are 1 and 2,
# node 1 has children 3 and 4, node 3 has children 5 and 6. The target's
# prediction at a node is accepted when it equals a child's token; the output
# carries the target's predictions along the accepted path.
_WRONG = 7777
_TREE_CASES = [
    pytest.param(
        [5000] * 7, 0, [5000, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], id="reject-all"
    ),
    pytest.param(
        [1000, 1002, _WRONG, 1004, _WRONG, 4242, _WRONG],
        3,
        [1000, 1002, 0, 1004, 0, 4242, 0],
        [0, 1, 3, 5],
        id="accept-all",
    ),
    pytest.param(
        [1000, _WRONG, _WRONG, _WRONG, _WRONG, _WRONG, _WRONG],
        1,
        [1000, _WRONG, 0, 0, 0, 0, 0],
        [0, 1, 0, 0],
        id="partial",
    ),
    pytest.param(
        [1000, 1003, _WRONG, _WRONG, 4242, _WRONG, _WRONG],
        2,
        [1000, 1003, 0, 0, 4242, 0, 0],
        [0, 1, 4, 0],
        id="mid-sibling",
    ),
]


@pytest.mark.parametrize("target_rows,accepted,predicts_out,indices", _TREE_CASES)
def test_verify_tree_greedy_acceptance_paths(
    kernel, target_rows, accepted, predicts_out, indices
):
    tree = _build_tree(kernel, 2, [-1, 0, 2])
    predicts = torch.zeros(2, 7, dtype=torch.int64, device="cuda")
    accept_index = torch.zeros(2, 4, dtype=torch.int64, device="cuda")
    accept_token_num = torch.zeros(2, dtype=torch.int64, device="cuda")
    target = torch.tensor([target_rows] * 2, device="cuda")

    kernel.verify_tree_greedy(
        predicts,
        accept_index,
        accept_token_num,
        tree[5].view(2, 7),
        retrive_index=tree[2],
        retrive_next_token=tree[3],
        retrive_next_sibling=tree[4],
        target_predict=target,
    )
    torch.cuda.synchronize()
    assert accept_token_num.tolist() == [accepted] * 2
    assert predicts[0].tolist() == predicts_out
    assert accept_index[0].tolist() == indices
