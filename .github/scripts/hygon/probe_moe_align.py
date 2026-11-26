#!/usr/bin/env python3
"""TEMP experiment (validation only, never merge): can FlagGems
``moe_align_block_size`` replace the ``sgl_kernel`` one on HCU?

SGLang's Triton fused_experts path calls
``sglang.kernels.ops.moe.moe_align_block_size`` (an AOT sgl_kernel op) before
launching; on HCU sgl_kernel is the plugin's placeholder, so every MoE model
fails with ``sgl_kernel.moe_align_block_size is unavailable``.

Round 5. Earlier rounds established that the FlagGems kernel is correct for
small inputs but wrong from numel 16384, where it switches to its TLE path.

Step 1 runs the FlagGems kernel as shipped against a plain Python
reference and compares per expert (what fused MoE actually needs).
Step 2 installs the integration hook's replacement at the symbol SGLang
calls and checks the buffers SGLang itself allocated.
Step 3 swaps SGLang's speculative ``fast_topk`` for SGLang's own pure-torch
one and exercises the MTP draft selection that calls it.
"""

from __future__ import annotations

import json
import sys
import time
import traceback

import torch

PREFIX = "[moe-probe]"
FLAG_GEMS_ALIGN_MODULE = "flag_gems.fused.moe_align_block_size"


def say(*parts: object) -> None:
    print(PREFIX, *parts, flush=True)


def reference(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    flat = topk_ids.flatten().tolist()
    total = len(flat)
    buckets: list[list[int]] = [[] for _ in range(num_experts)]
    for index, expert in enumerate(flat):
        if not 0 <= expert < num_experts:
            raise ValueError(f"expert id {expert} outside [0, {num_experts})")
        buckets[expert].append(index)
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for expert, bucket in enumerate(buckets):
        used = len(bucket)
        padded = (used + block_size - 1) // block_size * block_size
        sorted_ids += bucket + [total] * (padded - used)
        expert_ids += [expert] * (padded // block_size)
    return (
        torch.tensor(sorted_ids, dtype=torch.int32),
        torch.tensor(expert_ids, dtype=torch.int32),
        len(sorted_ids),
        total,
    )


def compare(label, block_size, sorted_ids, expert_ids, num_post_pad, ref) -> bool:
    """Return whether the output is SEMANTICALLY correct.

    The valid prefix is num_tokens_post_pad entries (buffers are allocated to
    the upper bound). A fused-MoE kernel only needs: the block -> expert map,
    every token landing in a block of its own expert, and the padding of each
    expert at the END of that expert's region. Which of an expert's blocks a
    token lands in does not matter, so tokens are compared per expert; the
    strict per-block equality is reported as information only.
    """
    ref_ids, ref_expert_ids, ref_num, pad_value = ref
    used = int(num_post_pad.reshape(-1)[0])
    if used % block_size:
        say(f"  {label}: num_tokens_post_pad {used} not a multiple of {block_size}")
        return False
    blocks = used // block_size
    if (
        blocks * block_size > sorted_ids.reshape(-1).numel()
        or blocks > expert_ids.reshape(-1).numel()
    ):
        say(f"  {label}: num_tokens_post_pad {used} exceeds the returned buffers")
        return False
    got_expert_ids = expert_ids.reshape(-1)[:blocks].tolist()
    problems = []
    if used != ref_num:
        problems.append(f"num_tokens_post_pad {used} != {ref_num}")
    if got_expert_ids != ref_expert_ids[:blocks].tolist():
        problems.append("block->expert map differs")
        return False
    flat = sorted_ids.reshape(-1)[:used].tolist()
    ref_flat = ref_ids.tolist()
    strict_bad = 0
    per_expert_bad = []
    tail_bad = []
    expert = got_expert_ids[0]
    region: list[int] = []
    ref_region: list[int] = []
    regions: list[tuple[int, list[int], list[int]]] = []

    def flush() -> None:
        if region:
            regions.append((expert, region, ref_region))

    for index in range(used):
        current_expert = got_expert_ids[index // block_size]
        if current_expert != expert:
            flush()
            expert, region, ref_region = current_expert, [], []
        region.append(flat[index])
        ref_region.append(ref_flat[index])
    flush()
    for expert, got_region, want_region in regions:
        got_real = sorted(v for v in got_region if v != pad_value)
        want_real = sorted(v for v in want_region if v != pad_value)
        if got_real != want_real:
            per_expert_bad.append(expert)
        got_pads = sum(1 for v in got_region if v == pad_value)
        want_pads = sum(1 for v in want_region if v == pad_value)
        if (
            got_pads != want_pads
            or got_region[len(got_region) - got_pads :] != [pad_value] * got_pads
        ):
            tail_bad.append(expert)
        got_blocks = [
            sorted(got_region[i : i + block_size])
            for i in range(0, len(got_region), block_size)
        ]
        want_blocks = [
            sorted(want_region[i : i + block_size])
            for i in range(0, len(want_region), block_size)
        ]
        strict_bad += sum(1 for g, w in zip(got_blocks, want_blocks) if g != w)
    if per_expert_bad:
        problems.append(f"token sets differ for experts {per_expert_bad[:5]}")
    if tail_bad:
        problems.append(f"padding not at region end for experts {tail_bad[:5]}")
    out_of_range = sum(1 for v in flat if v >= pad_value and v != pad_value)
    if out_of_range:
        problems.append(f"{out_of_range} entries are neither a token nor the pad value")
    if problems:
        say(f"  {label}: {'; '.join(problems)}")
    else:
        say(
            f"  {label}: SEMANTIC OK"
            + (
                f" (order differs from the reference in {strict_bad} of {blocks} blocks)"
                if strict_bad
                else ""
            )
        )
    return not problems


def make_case(tokens, topk, experts):
    generator = torch.Generator(device="cpu").manual_seed(tokens * 1000 + experts)
    return (
        torch.randint(
            0, experts, (tokens, topk), dtype=torch.int32, generator=generator
        )
    ).to("cuda")


def step1_flag_gems_kernel() -> bool:
    say("=== step 1: FlagGems moe_align_block_size as shipped vs reference ===")
    import flag_gems  # noqa: F401

    say("  module HAS_TLE as shipped:", sys.modules[FLAG_GEMS_ALIGN_MODULE].HAS_TLE)
    cases = [
        (1, 8, 256, 64),
        (1, 8, 257, 64),
        (37, 8, 256, 16),
        (512, 8, 256, 64),
        (1000, 8, 257, 64),
        (1500, 8, 257, 64),
        (2046, 8, 257, 64),
        (2048, 8, 257, 64),
        (2050, 8, 257, 64),
        (3000, 8, 257, 64),
        (4096, 8, 256, 64),
        (4098, 8, 256, 64),
        (8192, 8, 256, 64),
        (16384, 8, 256, 64),
        (16384, 10, 512, 32),
    ]
    all_ok = True
    for tokens, topk, experts, block in cases:
        topk_ids = make_case(tokens, topk, experts)
        label = f"numel={tokens * topk} E={experts} block={block}"
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = flag_gems.moe_align_block_size(topk_ids, block, experts)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            ok = compare(
                label,
                block,
                out[0].cpu(),
                out[1].cpu(),
                out[2].cpu(),
                reference(topk_ids.cpu(), block, experts),
            )
            say(f"    ({elapsed:.1f} ms)")
            all_ok = all_ok and ok
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(f"  {label}: RAISED {type(error).__name__}: {str(error)[:300]}")
            traceback.print_exc()
            all_ok = False
    return all_ok


def load_hook_module():
    """The integration hook, loaded by path WITHOUT installing it."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("hcu_sgl_kernel_fallback_probe.py")
    spec = importlib.util.spec_from_file_location("hcu_fallback_probe_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step2_sglang_symbol() -> bool:
    say("=== step 2: drop-in at sglang.kernels.ops.moe.moe_align_block_size ===")
    import flag_gems  # noqa: F401
    import sglang.kernels.ops.moe  # noqa: F401
    import sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size  # noqa: F401

    # ``import a.b.c as x`` and the package attribute both resolve to the
    # FUNCTION of the same name here, so take the modules from sys.modules.
    kernels_moe = sys.modules["sglang.kernels.ops.moe"]
    runner = sys.modules[
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"
    ]
    say("  original symbol:", kernels_moe.moe_align_block_size)
    all_ok = True
    try:
        runner.moe_align_block_size(
            torch.randint(0, 256, (16, 8), dtype=torch.int32, device="cuda"), 64, 256
        )
    except Exception as error:  # noqa: BLE001 - the failure under investigation
        say(f"  unpatched call raises {type(error).__name__}: {str(error)[:200]}")
    else:
        say("  unpatched call unexpectedly succeeded; nothing to replace")
        return True

    hook = load_hook_module()
    drop_in = hook.moe_align_block_size
    # The SGLang wrapper calls the symbol it bound at import time, so the
    # caller module's own name has to be rebound as well.
    kernels_moe.moe_align_block_size = drop_in
    say("  caller bound name:", getattr(runner, "sgl_moe_align_block_size", None))
    if getattr(runner, "sgl_moe_align_block_size", None) is not None:
        runner.sgl_moe_align_block_size = drop_in
    for tokens in (1, 2, 16, 31, 77, 512, 2048, 4096, 8192):
        topk_ids = make_case(tokens, 8, 256)
        try:
            sorted_ids, expert_ids, num_post_pad = runner.moe_align_block_size(
                topk_ids, 64, 256
            )
            torch.cuda.synchronize()
            ok = compare(
                f"sglang wrapper tokens={tokens}",
                64,
                sorted_ids.cpu(),
                expert_ids.cpu(),
                num_post_pad.cpu(),
                reference(topk_ids.cpu(), 64, 256),
            )
            pad_value = int(topk_ids.numel())
            tail = sorted_ids[int(num_post_pad[0]) :]
            tail_ok = bool((tail == pad_value).all()) if tail.numel() else True
            say(
                f"  tokens={tokens}: {'OK' if ok and tail_ok else 'MISMATCH'} "
                f"tail_filled={tail_ok} post_pad={int(num_post_pad[0])}"
            )
            all_ok = all_ok and ok and tail_ok
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(f"  tokens={tokens}: RAISED {type(error).__name__}: {str(error)[:300]}")
            traceback.print_exc()
            all_ok = False
    return all_ok


def step3_fast_topk() -> bool:
    """MTP draft: spec_utils imports fast_topk from the sgl_kernel placeholder.

    Candidates at the exact symbol: torch.topk (same shapes for every k) and
    SGLang's own non-CUDA fallback (a keepdim max shortcut for k == 1). The
    mtp acceptance runs EAGLE with topk 1, so both k values are exercised.
    """
    say("=== step 3: fast_topk in sglang.srt.speculative.spec_utils ===")
    import sglang.srt.speculative.spec_utils  # noqa: F401
    from sglang.srt.utils.common import fast_topk as sglang_fallback_fast_topk

    hook = load_hook_module()
    spec_utils = sys.modules["sglang.srt.speculative.spec_utils"]
    say("  current symbol:", getattr(spec_utils, "fast_topk", None))
    try:
        spec_utils.fast_topk(torch.rand(4, 4, device="cuda"), 1, dim=-1)
    except Exception as error:  # noqa: BLE001 - the failure under investigation
        say(f"  unpatched call raises {type(error).__name__}: {str(error)[:200]}")
    else:
        say("  unpatched call unexpectedly succeeded; nothing to replace")
        return True

    all_ok = True
    # Shapes as eagle_worker_v2.draft_forward calls it for i >= 1: topk_p and
    # topk_index carry topk*topk candidates per request, scores is the current
    # step's topk scores per request, hidden has one row per topk candidate.
    batch, width = 8, 16
    for candidate_name, candidate in (
        ("torch.topk", hook.fast_topk),
        ("sglang fallback", sglang_fallback_fast_topk),
    ):
        spec_utils.fast_topk = candidate
        say(f"  candidate: {candidate_name}")
        for topk in (1, 2, 4):
            generator = torch.Generator(device="cpu").manual_seed(topk)
            topk_p = torch.rand(batch, topk * topk, generator=generator).to("cuda")
            topk_index = (
                torch.arange(batch * topk * topk, dtype=torch.int64)
                .view(batch, topk * topk)
                .to("cuda")
            )
            hidden = torch.rand(batch * topk, width, generator=generator).to("cuda")
            scores = torch.rand(batch, topk, generator=generator).to("cuda")
            try:
                torch._dynamo.reset()
                input_ids, hidden_out, probs, tree_info = (
                    spec_utils._select_top_k_tokens_later(
                        1, topk_p, topk_index, hidden, scores, topk
                    )
                )
                torch.cuda.synchronize()
            except Exception as error:  # noqa: BLE001 - probe reports and continues
                say(
                    f"    topk={topk}: RAISED {type(error).__name__}: {str(error)[:200]}"
                )
                all_ok = all_ok and False
                continue
            expect = (
                (scores.unsqueeze(2) * topk_p.view(-1, topk, topk))
                .flatten(start_dim=1)
                .topk(topk, dim=-1)
            )
            close = bool(
                torch.allclose(
                    probs.reshape(batch, -1).float().cpu(),
                    expect.values.cpu(),
                    atol=1e-5,
                )
            )
            index_match = bool(
                torch.equal(
                    torch.sort(
                        torch.gather(
                            topk_index.cpu().view(batch, -1), 1, expect.indices.cpu()
                        ).flatten()
                    ).values,
                    torch.sort(input_ids.cpu()).values,
                )
            )
            shapes_ok = (
                tuple(input_ids.shape) == (batch * topk,)
                and tuple(hidden_out.shape) == (batch * topk, width)
                and tuple(probs.shape) == (batch, topk)
            )
            say(
                f"    topk={topk}: probs_match_torch={close} "
                f"selected_tokens_match_torch={index_match} shapes_ok={shapes_ok} "
                f"probs_shape={tuple(probs.shape)}"
            )
            all_ok = all_ok and close and index_match and shapes_ok
    return all_ok


# SGLang passes the tree tensors to verify_tree_greedy under these keywords.
TREE_KEYS = (
    "retr" + "ive_index",
    "retr" + "ive_next_token",
    "retr" + "ive_next_sibling",
    "target_predict",
)


def step4_round2_table() -> bool:
    """The operators after moe_align_block_size on the 35B-A3B and mtp paths.

    Round 1 got past moe_align and raised ``sgl_kernel.silu_and_mul`` inside
    Triton fused_experts; the mtp path additionally needs the tree builder and
    greedy verifier. Each is checked against a plain torch reference on device,
    through the symbol SGLang's own module has bound.
    """
    say("=== step 4: silu_and_mul / moe_sum_reduce / tree build+verify ===")
    hook = load_hook_module()
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe
    import sglang.srt.speculative.eagle_utils as eagle_utils

    say(
        "  eagle_utils.sgl_build_tree_kernel_efficient is hook:",
        eagle_utils.sgl_build_tree_kernel_efficient is hook.build_tree_kernel_efficient,
    )
    all_ok = True
    device = "cuda"

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        say(f"  {label}: {'ok' if ok else 'MISMATCH'} {detail}")
        all_ok = all_ok and ok

    say(
        "  fused_moe.silu_and_mul is hook:", fused_moe.silu_and_mul is hook.silu_and_mul
    )

    # silu_and_mul(x, out): silu(x[:, :d]) * x[:, d:]
    rows, hidden = 257, 512
    x = torch.randn(rows, 2 * hidden, device=device, dtype=torch.bfloat16)
    out = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)
    fused_moe.silu_and_mul(x, out)
    ref = torch.nn.functional.silu(x[:, :hidden].float()) * x[:, hidden:].float()
    check(
        "silu_and_mul",
        torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2),
        f"max_abs_diff={(out.float() - ref).abs().max().item():.4g}",
    )

    # moe_sum_reduce_triton(x, out, scale): sum over topk * scale
    tokens, topk = 64, 8
    x = torch.randn(tokens, topk, hidden, device=device, dtype=torch.bfloat16)
    out = torch.empty(tokens, hidden, device=device, dtype=torch.bfloat16)
    hook.moe_sum_reduce(x, out, 0.5)
    ref = x.float().sum(1) * 0.5
    check(
        "moe_sum_reduce",
        torch.allclose(out.float(), ref, atol=2e-1, rtol=2e-2),
        f"max_abs_diff={(out.float() - ref).abs().max().item():.4g}",
    )

    # Tree: topk 1 chain (the mtp acceptance shape) and topk 2, depth 3.
    for topk in (1, 2):
        steps, batch, prompt = 3, 2, 7
        num_draft = 1 + steps * topk
        parent_list = torch.zeros(batch, steps * topk, device=device, dtype=torch.long)
        if topk > 1:
            parent_list = torch.arange(steps * topk, device=device) // topk
            parent_list = parent_list.expand(batch, -1).contiguous()
        scores_index = (
            torch.arange(batch * num_draft, device=device, dtype=torch.long)
            .view(batch, -1)[:, 1:]
            .contiguous()
        )
        bonus = torch.arange(100, 100 + batch, device=device, dtype=torch.long)
        draft_tokens = torch.arange(
            1000, 1000 + batch * steps * topk, device=device, dtype=torch.long
        ).view(batch, steps * topk)
        try:
            (
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens_all,
            ) = eagle_utils.build_tree_kernel_efficient(
                bonus,
                parent_list,
                scores_index,
                draft_tokens,
                torch.full((batch,), prompt, device=device, dtype=torch.long),
                batch * prompt,
                topk,
                steps,
                num_draft,
            )
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            check(
                f"build_tree topk={topk}",
                False,
                f"RAISED {type(error).__name__}: {str(error)[:300]}",
            )
            continue
        draft_tokens_all = draft_tokens_all.view(batch, num_draft)
        # (a) nothing accepted: the target disagrees at the root.
        predicts = torch.zeros(batch, num_draft, device=device, dtype=torch.long)
        accept_index = torch.zeros(batch, steps, device=device, dtype=torch.long)
        accept_num = torch.zeros(batch, device=device, dtype=torch.long)
        target_predict = torch.full(
            (batch, num_draft), -1, device=device, dtype=torch.long
        )
        target_predict[:, 0] = 5000 + torch.arange(batch, device=device)
        try:
            hook.verify_tree_greedy(
                predicts,
                accept_index,
                accept_num,
                draft_tokens_all,
                **dict(
                    zip(
                        TREE_KEYS,
                        (
                            retrieve_index,
                            retrieve_next_token,
                            retrieve_next_sibling,
                            target_predict,
                        ),
                    )
                ),
            )
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            check(
                f"verify_tree_greedy topk={topk}",
                False,
                f"RAISED {type(error).__name__}: {str(error)[:300]}",
            )
            continue
        check(
            f"verify reject-all topk={topk}",
            bool((accept_num == 0).all())
            and torch.equal(predicts[:, 0], target_predict[:, 0]),
            f"accept_num={accept_num.tolist()} predicts0={predicts[:, 0].tolist()}",
        )
        # (b) chain fully accepted (topk 1): the target repeats every draft token.
        if topk == 1:
            predicts.zero_()
            accept_index.zero_()
            accept_num.zero_()
            target_predict = draft_tokens_all.clone()
            hook.verify_tree_greedy(
                predicts,
                accept_index,
                accept_num,
                draft_tokens_all,
                **dict(
                    zip(
                        TREE_KEYS,
                        (
                            retrieve_index,
                            retrieve_next_token,
                            retrieve_next_sibling,
                            target_predict,
                        ),
                    )
                ),
            )
            torch.cuda.synchronize()
            check(
                "verify accept-all topk=1",
                bool((accept_num == steps).all())
                and torch.equal(
                    predicts[:, : steps + 1], draft_tokens_all[:, : steps + 1]
                ),
                f"accept_num={accept_num.tolist()} predicts={predicts[0].tolist()} draft={draft_tokens_all[0].tolist()} accept_index={accept_index[0].tolist()}",
            )
    return all_ok


def step5_paged_attention_head_dim() -> bool:
    """FlagSVT's fa256.py (2026-11-26): the vendor flash_attn of the torch 2.9
    runtime returned all-zero vectors for paged decode at head_dim 256 on
    Qwen3.6-27B (token id 0 after the first token). Same tensors and the same
    pass mark here: paged decode against hand-written attention, max_diff
    < 0.05.
    """
    say("=== step 5: flash_attn paged decode x head_dim (FlagSVT fa256.py) ===")
    try:
        import flash_attn
    except Exception as error:  # noqa: BLE001 - probe reports and continues
        say(f"  flash_attn import failed: {error!r}")
        return False
    say("  flash_attn", getattr(flash_attn, "__version__", "?"))

    all_ok = True
    device = "cuda"
    for head_dim in (64, 128, 192, 224, 256):
        heads, kv_heads, seqlen, page = 24, 4, 64, 64
        generator = torch.Generator(device="cpu").manual_seed(head_dim)
        k = torch.randn(1, seqlen, kv_heads, head_dim, generator=generator).to(
            device, torch.bfloat16
        )
        v = torch.randn(k.shape, generator=generator).to(device, torch.bfloat16)
        k_cache = torch.zeros(
            1, page, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        v_cache = torch.zeros_like(k_cache)
        k_cache[0, :seqlen] = k[0]
        v_cache[0, :seqlen] = v[0]
        q = torch.randn(1, 1, heads, head_dim, generator=generator).to(
            device, torch.bfloat16
        )
        out = flash_attn.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=torch.tensor([seqlen], dtype=torch.int32, device=device),
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
            softmax_scale=head_dim**-0.5,
            causal=True,
        )
        group = heads // kv_heads
        qf = q[0, 0].float().view(kv_heads, group, head_dim)
        kf = k[0].float().permute(1, 0, 2)  # (kv_heads, seqlen, head_dim)
        vf = v[0].float().permute(1, 0, 2)
        scores = torch.einsum("hgd,hsd->hgs", qf, kf) * head_dim**-0.5
        ref = torch.einsum("hgs,hsd->hgd", scores.softmax(-1), vf).reshape(
            heads, head_dim
        )
        diff = (out[0, 0].float() - ref).abs().max().item()
        ok = out.float().abs().sum().item() > 0 and diff < 0.05
        say(
            f"  head_dim={head_dim}: paged_abs_sum={out.float().abs().sum().item():.4g} "
            f"max_diff_vs_manual={diff:.4g} {'OK' if ok else 'BROKEN'}"
        )
        all_ok = all_ok and ok
    return all_ok


def main() -> int:
    say(
        "torch",
        torch.__version__,
        "cuda available",
        torch.cuda.is_available(),
        "gpus",
        torch.cuda.device_count(),
    )
    for path in (
        "/models/Qwen3.6-35B-A3B/config.json",
        "/models/Qwen3.6-35B-A3B/config.json",
    ):
        try:
            with open(path) as handle:
                config = json.load(handle)
            keys = (
                "architectures",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "norm_topk_prob",
            )
            say(path, {key: config.get(key) for key in keys})
        except OSError as error:
            say(path, "unreadable:", error)
    results = {}
    for name, step in (
        ("step1", step1_flag_gems_kernel),
        ("step2", step2_sglang_symbol),
        ("step3", step3_fast_topk),
        ("step4", step4_round2_table),
        ("step5", step5_paged_attention_head_dim),
    ):
        try:
            results[name] = step()
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(f"{name} RAISED {type(error).__name__}: {str(error)[:400]}")
            traceback.print_exc()
            results[name] = False
    say("RESULT", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
