#!/usr/bin/env python3
"""TEMP experiment (validation only, never merge): can FlagGems
``moe_align_block_size`` replace the ``sgl_kernel`` one on HCU?

SGLang's Triton fused_experts path calls
``sglang.kernels.ops.moe.moe_align_block_size`` (an AOT sgl_kernel op) before
launching; on HCU sgl_kernel is the plugin's placeholder, so every MoE model
fails with ``sgl_kernel.moe_align_block_size is unavailable``.

Step 1 runs the FlagGems kernel against a plain Python reference for decode-
and prefill-shaped inputs. Step 2 installs the drop-in replacement at the
exact symbol SGLang calls and checks the buffers SGLang itself allocated.
"""

from __future__ import annotations

import json
import traceback

import torch

PREFIX = "[moe-probe]"


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
    )


def compare(label, block_size, out, ref) -> bool:
    sorted_ids, expert_ids, num_post_pad = out
    ref_ids, ref_expert_ids, ref_num = ref
    ok = True
    if int(num_post_pad.reshape(-1)[0]) != ref_num:
        say(
            f"  {label}: num_tokens_post_pad {int(num_post_pad.reshape(-1)[0])} != {ref_num}"
        )
        ok = False
    if expert_ids.tolist() != ref_expert_ids.tolist():
        say(f"  {label}: expert_ids differ")
        ok = False
    blocks = sorted_ids.reshape(-1, block_size).tolist()
    ref_blocks = ref_ids.reshape(-1, block_size).tolist()
    if len(blocks) != len(ref_blocks):
        say(f"  {label}: {len(blocks)} blocks != {len(ref_blocks)}")
        return False
    bad = [
        index
        for index, (got, want) in enumerate(zip(blocks, ref_blocks))
        if sorted(got) != sorted(want)
    ]
    if bad:
        say(f"  {label}: {len(bad)} blocks differ, first {bad[:3]}")
        ok = False
    return ok


def step1_flag_gems_kernel() -> bool:
    say("=== step 1: FlagGems moe_align_block_size vs reference ===")
    import flag_gems

    fn = getattr(flag_gems, "moe_align_block_size", None)
    if fn is None:
        say("flag_gems has no moe_align_block_size")
        return False
    device = "cuda"
    cases = [
        (1, 8, 256, 64),
        (1, 8, 257, 64),
        (4, 8, 256, 64),
        (64, 8, 256, 64),
        (512, 8, 256, 64),
        (4096, 8, 256, 64),
        (1, 10, 128, 32),
        (37, 8, 256, 16),
        (2048, 8, 257, 64),
    ]
    all_ok = True
    for tokens, topk, experts, block in cases:
        generator = torch.Generator(device="cpu").manual_seed(tokens * 1000 + experts)
        topk_ids = torch.randint(
            0, experts, (tokens, topk), dtype=torch.int32, generator=generator
        ).to(device)
        try:
            import time

            start = time.perf_counter()
            out = fn(topk_ids, block, experts)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            out = tuple(part.cpu() for part in out)
            ok = compare(
                f"tokens={tokens} topk={topk} E={experts} block={block}",
                block,
                out,
                reference(topk_ids.cpu(), block, experts),
            )
            say(
                f"  tokens={tokens} topk={topk} E={experts} block={block}: "
                f"{'OK' if ok else 'MISMATCH'} ({elapsed:.1f} ms, "
                f"dtypes={[part.dtype for part in out]})"
            )
            all_ok = all_ok and ok
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(
                f"  tokens={tokens} topk={topk} E={experts} block={block}: "
                f"RAISED {type(error).__name__}: {str(error)[:300]}"
            )
            traceback.print_exc()
            all_ok = False
    return all_ok


def make_drop_in(flag_gems_fn):
    def moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids=False,
        ignore_invalid_expert=False,
    ):
        sorted_ids, expert_ids, num_post_pad = flag_gems_fn(
            topk_ids, block_size, num_experts
        )
        used = int(num_post_pad.reshape(-1)[0])
        if used > sorted_token_ids.numel():
            raise RuntimeError(
                f"FlagGems produced {used} sorted ids, SGLang allocated "
                f"{sorted_token_ids.numel()}"
            )
        sorted_token_ids.fill_(topk_ids.numel())
        sorted_token_ids[:used].copy_(sorted_ids[:used])
        experts_ids[: len(expert_ids)].copy_(expert_ids)
        num_tokens_post_pad.copy_(num_post_pad.reshape(num_tokens_post_pad.shape))

    return moe_align_block_size


def step2_sglang_symbol() -> bool:
    say("=== step 2: drop-in at sglang.kernels.ops.moe.moe_align_block_size ===")
    import flag_gems
    import sglang.kernels.ops.moe as kernels_moe
    import sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size as runner

    say("  original symbol:", kernels_moe.moe_align_block_size)
    try:
        topk_ids = torch.randint(0, 256, (16, 8), dtype=torch.int32, device="cuda")
        runner.moe_align_block_size(topk_ids, 64, 256)
    except Exception as error:  # noqa: BLE001 - the failure under investigation
        say(f"  unpatched call raises {type(error).__name__}: {str(error)[:200]}")
    else:
        say("  unpatched call unexpectedly succeeded; nothing to replace")
        return True

    kernels_moe.moe_align_block_size = make_drop_in(flag_gems.moe_align_block_size)
    all_ok = True
    for tokens in (1, 16, 77, 512, 2048):
        topk_ids = torch.randint(0, 256, (tokens, 8), dtype=torch.int32, device="cuda")
        sorted_ids, expert_ids, num_post_pad = runner.moe_align_block_size(
            topk_ids, 64, 256
        )
        ok = compare(
            f"sglang wrapper tokens={tokens}",
            64,
            (sorted_ids.cpu(), expert_ids.cpu(), num_post_pad.cpu()),
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
        "/models/Qwen3.6-27B/config.json",
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
    ):
        try:
            results[name] = step()
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(f"{name} RAISED {type(error).__name__}: {str(error)[:400]}")
            traceback.print_exc()
            results[name] = False
    try:
        results["step3"] = step3_fast_topk()
    except Exception as error:  # noqa: BLE001 - probe reports and continues
        say(f"step3 RAISED {type(error).__name__}: {str(error)[:400]}")
        traceback.print_exc()
        results["step3"] = False
    say("RESULT", results)
    return 0


def step3_fast_topk() -> bool:
    """MTP draft: spec_utils imports fast_topk from the sgl_kernel placeholder.

    SGLang's own non-CUDA/HIP branch uses a pure-torch fallback
    (sglang.srt.utils.common.fast_topk); try it at the exact symbol.
    """
    say("=== step 3: fast_topk in sglang.srt.speculative.spec_utils ===")
    import sglang.srt.speculative.spec_utils as spec_utils
    from sglang.srt.utils.common import fast_topk as torch_fast_topk

    say("  current symbol:", getattr(spec_utils, "fast_topk", None))
    batch, topk = 4, 2
    scores = torch.rand(batch, topk * topk, device="cuda")
    try:
        spec_utils.fast_topk(scores, topk, dim=-1)
    except Exception as error:  # noqa: BLE001 - the failure under investigation
        say(f"  unpatched call raises {type(error).__name__}: {str(error)[:200]}")
    else:
        say("  unpatched call unexpectedly succeeded; nothing to replace")
        return True

    spec_utils.fast_topk = torch_fast_topk
    topk_p = torch.rand(8, topk, device="cuda")
    topk_index = torch.arange(8 * topk, device="cuda", dtype=torch.int64).view(8, topk)
    hidden = torch.rand(8, 16, device="cuda")
    scores_full = torch.rand(8, topk * 64, device="cuda")
    all_ok = True
    for iteration in (1, 2):
        try:
            input_ids, hidden_out, probs, tree_info = (
                spec_utils._select_top_k_tokens_later(
                    iteration, topk_p, topk_index, hidden, scores_full, topk
                )
            )
        except Exception as error:  # noqa: BLE001 - probe reports and continues
            say(
                f"  iteration={iteration} RAISED {type(error).__name__}: {str(error)[:300]}"
            )
            traceback.print_exc()
            all_ok = False
            continue
        expect_probs = (
            (scores_full.unsqueeze(2) * topk_p.view(-1, topk, topk))
            .flatten(start_dim=1)
            .topk(topk, dim=-1)
            .values
        )
        close = bool(torch.allclose(probs, expect_probs, atol=1e-6))
        say(
            f"  iteration={iteration}: input_ids={tuple(input_ids.shape)} "
            f"hidden={tuple(hidden_out.shape)} probs={tuple(probs.shape)} "
            f"topk_probs_match_torch={close}"
        )
        all_ok = all_ok and close
    return all_ok


if __name__ == "__main__":
    raise SystemExit(main())
