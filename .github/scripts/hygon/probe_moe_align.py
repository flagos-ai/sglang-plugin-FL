#!/usr/bin/env python3
"""TEMP experiment (validation only, never merge): can FlagGems
``moe_align_block_size`` replace the ``sgl_kernel`` one on HCU?

SGLang's Triton fused_experts path calls
``sglang.kernels.ops.moe.moe_align_block_size`` (an AOT sgl_kernel op) before
launching; on HCU sgl_kernel is the plugin's placeholder, so every MoE model
fails with ``sgl_kernel.moe_align_block_size is unavailable``.

Round 3. Earlier rounds established that the FlagGems kernel is correct for
small inputs but wrong from numel 16384, where it switches to its TLE path.

Step 1 runs the FlagGems kernel as shipped and with its TLE switch off,
against a plain Python reference, across the switch-over sizes.
Step 2 installs the drop-in replacement at the exact symbol SGLang calls
(with the TLE switch off) and checks the buffers SGLang itself allocated.
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
    )


def compare(label, block_size, sorted_ids, expert_ids, num_post_pad, ref) -> bool:
    """Compare the VALID prefix: align kernels return buffers allocated to the
    upper bound; only the first num_tokens_post_pad entries are meaningful."""
    ref_ids, ref_expert_ids, ref_num = ref
    used = int(num_post_pad.reshape(-1)[0])
    if used % block_size:
        say(f"  {label}: num_tokens_post_pad {used} not a multiple of {block_size}")
        return False
    ok = True
    if used != ref_num:
        say(f"  {label}: num_tokens_post_pad {used} != {ref_num}")
        ok = False
    blocks = used // block_size
    got_expert_ids = expert_ids.reshape(-1)[:blocks].tolist()
    if got_expert_ids != ref_expert_ids[:blocks].tolist():
        first = next(
            (
                i
                for i, (got, want) in enumerate(
                    zip(got_expert_ids, ref_expert_ids.tolist())
                )
                if got != want
            ),
            None,
        )
        say(f"  {label}: expert_ids differ (first mismatch at block {first})")
        ok = False
    got = sorted_ids.reshape(-1)[:used].reshape(blocks, block_size).tolist()
    want = ref_ids[: blocks * block_size].reshape(blocks, block_size).tolist()
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if sorted(g) != sorted(w)]
    if bad:
        say(f"  {label}: {len(bad)} of {blocks} blocks differ, first {bad[:3]}")
        first = bad[0]
        say(f"    block {first}: expert {got_expert_ids[first]}")
        say(
            f"      got  real={[v for v in got[first] if v != int(ref_num)][:12]} "
            f"pads={sum(1 for v in got[first] if v == int(ref_num))}"
        )
        say(
            f"      want real={[v for v in want[first] if v != int(ref_num)][:12]} "
            f"pads={sum(1 for v in want[first] if v == int(ref_num))}"
        )
        all_got = sorted(v for row in got for v in row if v != int(ref_num))
        all_want = sorted(v for row in want for v in row if v != int(ref_num))
        say(
            f"    same multiset of real tokens overall: {all_got == all_want} "
            f"(got {len(all_got)}, want {len(all_want)})"
        )
        ok = False
    return ok


def align_via_flag_gems(topk_ids, block_size, num_experts, *, tle: bool):
    """FlagGems' public kernel; ``tle=False`` forces its classic 4-stage path."""
    import flag_gems

    module = sys.modules[FLAG_GEMS_ALIGN_MODULE]
    saved = module.HAS_TLE
    module.HAS_TLE = bool(tle)
    try:
        return flag_gems.moe_align_block_size(topk_ids, block_size, num_experts)
    finally:
        module.HAS_TLE = saved


def make_case(tokens, topk, experts):
    generator = torch.Generator(device="cpu").manual_seed(tokens * 1000 + experts)
    return (
        torch.randint(
            0, experts, (tokens, topk), dtype=torch.int32, generator=generator
        )
    ).to("cuda")


def step1_flag_gems_kernel() -> bool:
    say("=== step 1: FlagGems moe_align_block_size (as shipped vs TLE off) ===")
    import flag_gems  # noqa: F401

    say(
        "  module HAS_TLE as shipped:",
        sys.modules[FLAG_GEMS_ALIGN_MODULE].HAS_TLE
        if FLAG_GEMS_ALIGN_MODULE in sys.modules
        else "not imported yet",
    )
    cases = [
        (1, 8, 256, 64),
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
    ]
    results = {}
    for tokens, topk, experts, block in cases:
        topk_ids = make_case(tokens, topk, experts)
        ref = reference(topk_ids.cpu(), block, experts)
        for tle in (True, False):
            label = f"numel={tokens * topk} E={experts + 0} block={block} tle={'on' if tle else 'off'}"
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = align_via_flag_gems(topk_ids, block, experts, tle=tle)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                ok = compare(
                    label, block, out[0].cpu(), out[1].cpu(), out[2].cpu(), ref
                )
                say(f"  {label}: {'OK' if ok else 'MISMATCH'} ({elapsed:.1f} ms)")
            except Exception as error:  # noqa: BLE001 - probe reports and continues
                say(f"  {label}: RAISED {type(error).__name__}: {str(error)[:300]}")
                traceback.print_exc()
                ok = False
            results[(tokens, tle)] = ok
    return all(results[(tokens, False)] for tokens, _, _, _ in cases)


def make_drop_in():
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
        if ignore_invalid_expert:
            raise RuntimeError("probe drop-in does not implement ignore_invalid_expert")
        sorted_ids, expert_ids, num_post_pad = align_via_flag_gems(
            topk_ids, block_size, num_experts, tle=False
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

    kernels_moe.moe_align_block_size = make_drop_in()
    for tokens in (1, 16, 77, 512, 2048, 4096, 8192):
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

    SGLang's own non-CUDA/HIP branch uses a pure-torch fallback
    (sglang.srt.utils.common.fast_topk); try it at the exact symbol.
    """
    say("=== step 3: fast_topk in sglang.srt.speculative.spec_utils ===")
    import sglang.srt.speculative.spec_utils  # noqa: F401
    from sglang.srt.utils.common import fast_topk as torch_fast_topk

    spec_utils = sys.modules["sglang.srt.speculative.spec_utils"]
    say("  current symbol:", getattr(spec_utils, "fast_topk", None))
    scores = torch.rand(4, 4, device="cuda")
    try:
        spec_utils.fast_topk(scores, 2, dim=-1)
    except Exception as error:  # noqa: BLE001 - the failure under investigation
        say(f"  unpatched call raises {type(error).__name__}: {str(error)[:200]}")
    else:
        say("  unpatched call unexpectedly succeeded; nothing to replace")
        return True

    spec_utils.fast_topk = torch_fast_topk
    # Shapes as eagle_worker_v2.draft_forward calls it for i >= 1: topk_p,
    # topk_index and scores carry topk*topk candidates per request, hidden is
    # one row per request.
    batch, topk = 8, 2
    generator = torch.Generator(device="cpu").manual_seed(0)
    topk_p = torch.rand(batch, topk * topk, generator=generator).to("cuda")
    topk_index = (
        torch.arange(batch * topk * topk, dtype=torch.int64)
        .view(batch, topk * topk)
        .to("cuda")
    )
    hidden = torch.rand(batch, 16, generator=generator).to("cuda")
    scores_full = torch.rand(batch, topk * topk, generator=generator).to("cuda")
    all_ok = True
    for iteration in (1, 2):
        try:
            input_ids, hidden_out, probs, _tree_info = (
                spec_utils._select_top_k_tokens_later(
                    iteration, topk_p, topk_index, hidden, scores_full, topk
                )
            )
            torch.cuda.synchronize()
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
        close = bool(torch.allclose(probs.float().cpu(), expect_probs.cpu(), atol=1e-5))
        shapes_ok = (
            tuple(input_ids.shape) == (batch * topk,)
            and tuple(hidden_out.shape) == (batch * topk, 16)
            and tuple(probs.shape) == (batch, topk)
        )
        say(
            f"  iteration={iteration}: input_ids={tuple(input_ids.shape)} "
            f"hidden={tuple(hidden_out.shape)} probs={tuple(probs.shape)} "
            f"topk_probs_match_torch={close} shapes_ok={shapes_ok}"
        )
        all_ok = all_ok and close and shapes_ok
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
