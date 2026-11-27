#!/usr/bin/env python3
"""TEMP experiment (validation only, never merge): does the vendor
``flash_attn.flash_attn_with_kvcache`` (paged decode) return correct values,
per head_dim / page_size / sequence length / batch / cache size?

FlagSVT reported that the torch-2.9 runtime returns whole-zero output for
``head_dim=256`` with no error (token id 0 from the second token on). This
probe compares the kernel against a plain ``softmax(q k^T / sqrt(d)) v``
written with ordinary torch ops, and prints one line per case. It never
fails the job: the log is the result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import sys
import traceback

import flash_attn
import torch

PREFIX = "[fa-probe]"
TOL = 0.05
# FA_PROBE_DEVICE / FA_PROBE_MOCK only exist to dry-run the probe on a CPU box.
DEV = os.environ.get("FA_PROBE_DEVICE", "cuda")
MOCK = os.environ.get("FA_PROBE_MOCK") == "1"


def say(*parts: object) -> None:
    print(PREFIX, *parts, flush=True)


def env_info() -> None:
    say("python", sys.version.split()[0])
    say("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
    try:
        say("flash_attn version", importlib.metadata.version("flash_attn"))
    except Exception as error:  # noqa: BLE001
        say("flash_attn version unavailable", repr(error))
    import flash_attn

    say("flash_attn file", getattr(flash_attn, "__file__", None))
    try:
        import flash_attn_interface

        say(
            "flash_attn_interface file", getattr(flash_attn_interface, "__file__", None)
        )
        say(
            "flash_attn_interface ops",
            sorted(
                n
                for n in dir(flash_attn_interface)
                if n.startswith("fwd") or n.startswith("bwd")
            )[:12],
        )
    except Exception as error:  # noqa: BLE001
        say("flash_attn_interface unavailable", repr(error))
    for name in sorted(os.environ):
        if any(
            key in name
            for key in ("FLASH", "HCU", "HIP_", "ROCM", "DTK", "TORCH_DEVICE")
        ):
            say("env", name, "=", os.environ[name])
    try:
        prop = torch.cuda.get_device_properties(0)
        say(
            "device",
            prop.name,
            "gcnArchName",
            getattr(prop, "gcnArchName", None),
            "count",
            torch.cuda.device_count(),
        )
    except Exception as error:  # noqa: BLE001
        say("device properties unavailable", repr(error))
    say(
        "uname",
        subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip(),
    )
    for path in ("/opt/hyhal/.info/version", "/usr/local/hyhal/.info/version"):
        if os.path.exists(path):
            say(path, open(path).read().strip().replace("\n", " | "))


def reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    table_row,
    length: int,
    scale: float,
):
    """q: (1, Hq, D); returns (1, Hq, D) in fp32. Query head h reads KV head h // group."""
    page = k_cache.shape[1]
    pages = (length + page - 1) // page
    slots = table_row[:pages].long()
    k = k_cache[slots].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:length]
    v = v_cache[slots].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:length]
    heads = q.shape[1]
    group = heads // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    score = torch.einsum("hd,shd->hs", q[0].float(), k.float()) * scale
    return torch.einsum("hs,shd->hd", score.softmax(dim=-1), v.float())[None]


def mock_paged_attention(q, k_cache, v_cache, table, lens, scale):
    """Plain-torch stand-in for the kernel; only used by the S0 self-check."""
    heads, dim = q.shape[2], q.shape[3]
    kv_heads = k_cache.shape[2]
    rows = []
    for row in range(q.shape[0]):
        flat = (
            table[row, : lens[row]].long()[:, None] * k_cache.shape[1]
            + torch.arange(k_cache.shape[1], device=q.device)
        ).reshape(-1)[: lens[row]]
        k = k_cache.reshape(-1, kv_heads, dim)[flat].float()
        v = v_cache.reshape(-1, kv_heads, dim)[flat].float()
        probs = (
            (
                q[row, 0].float().reshape(kv_heads, heads // kv_heads, 1, dim)
                * k.transpose(0, 1)[:, None]
            ).sum(-1)
            * scale
        ).softmax(-1)
        rows.append(
            (probs[:, :, None, :] @ v.transpose(0, 1)[:, None]).reshape(heads, dim)
        )
    return torch.stack(rows)[:, None]


def run_case(
    D,
    P,
    lengths,
    heads=(24, 4),
    layout="contig",
    num_blocks=None,
    num_splits=1,
    dtype=torch.bfloat16,
    seed=0,
    mock=False,
):
    """Return a dict describing one flash_attn_with_kvcache call against the reference."""
    torch.manual_seed(seed)
    dev = DEV
    batch = len(lengths)
    pages_needed = [(n + P - 1) // P for n in lengths]
    if num_blocks is None:
        num_blocks = sum(pages_needed) + 2 * batch + 1
    width = max(16, max(pages_needed))
    if layout == "contig":
        starts = [sum(pages_needed[:i]) for i in range(batch)]
    else:
        # Non-consecutive physical pages, the first one far from block 0.
        starts = [num_blocks - 1 - sum(pages_needed[i:]) - 2 * i for i in range(batch)]
    table = torch.zeros(batch, width, dtype=torch.int64, device=dev)
    for row in range(batch):
        table[row, : pages_needed[row]] = torch.arange(
            starts[row], starts[row] + pages_needed[row], device=dev
        )
        table[row, pages_needed[row] :] = (starts[row] + 3) % num_blocks
    table = table.to(torch.int32)
    k_cache = torch.randn(num_blocks, P, heads[1], D, dtype=dtype, device=dev)
    v_cache = torch.randn(num_blocks, P, heads[1], D, dtype=dtype, device=dev)
    q = torch.randn(batch, 1, heads[0], D, dtype=dtype, device=dev)
    lens = torch.tensor(lengths, dtype=torch.int32, device=dev)
    kwargs = {} if num_splits is None else {"num_splits": num_splits}
    if mock or MOCK:
        out = mock_paged_attention(q, k_cache, v_cache, table, lens, D**-0.5)
    else:
        out = flash_attn.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=lens,
            block_table=table,
            softmax_scale=D**-0.5,
            causal=True,
            softcap=0.0,
            **kwargs,
        )
    if DEV != "cpu":
        torch.cuda.synchronize()
    expected = torch.cat(
        [
            reference(q[b : b + 1, 0], k_cache, v_cache, table[b], lengths[b], D**-0.5)
            for b in range(batch)
        ]
    )
    actual = out[:, 0].float()
    diff = (actual - expected).abs()
    rows = actual.flatten(1).abs().sum(dim=1)
    return {
        "max_diff": diff.max().item(),
        "out_sum": actual.abs().sum().item(),
        "zero_rows": int((rows == 0).sum().item()),
        "rows": rows.numel(),
        "nan": bool(actual.isnan().any().item()),
        "inf": bool(actual.isinf().any().item()),
        "digest": hashlib.sha1(actual.cpu().numpy().tobytes()).hexdigest()[:10],
    }


def show(tag, D, P, lengths, heads, layout, num_blocks, num_splits, result, extra=""):
    status = (
        "OK"
        if (not result["nan"] and not result["inf"] and result["max_diff"] <= TOL)
        else "BROKEN"
    )
    kind = (
        "all-zero"
        if result["zero_rows"] == result["rows"]
        else ("partial-zero" if result["zero_rows"] else "nonzero")
    )
    say(
        f"{tag} D={D} P={P} S={lengths} B={len(lengths)} Hq/Hkv={heads[0]}/{heads[1]} layout={layout} "
        f"blocks={num_blocks} splits={num_splits} max_diff={result['max_diff']:.4g} "
        f"out_sum={result['out_sum']:.4g} zero_rows={result['zero_rows']}/{result['rows']} kind={kind} "
        f"nan={result['nan']} inf={result['inf']} digest={result['digest']} {status} {extra}",
    )


def guarded(tag, **kwargs):
    try:
        result = run_case(**kwargs)
    except Exception as error:  # noqa: BLE001
        say(f"{tag} {kwargs} EXCEPTION {type(error).__name__}: {str(error)[:200]}")
        return None
    keys = ("D", "P", "lengths", "heads", "layout", "num_blocks", "num_splits")
    say(f"RAW {tag} " + " ".join(f"{k}={kwargs.get(k)}" for k in keys))
    show(
        tag,
        *(
            kwargs.get(k, d)
            for k, d in zip(keys, (None, 64, None, (24, 4), "contig", None, 1))
        ),
        result,
    )
    return result


def main() -> None:
    env_info()
    lengths_by_s = {
        1: [1],
        2: [2],
        63: [63],
        64: [64],
        65: [65],
        127: [127],
        128: [128],
        129: [129],
        300: [300],
        2000: [2000],
    }

    try:
        # S0: self-check of the probe's own reference against a plain-torch stand-in.
        for case in (
            dict(D=64, P=64, lengths=[65]),
            dict(D=256, P=64, lengths=[2, 65, 3]),
            dict(D=256, P=128, lengths=[2000], heads=(12, 2), layout="scatter"),
            dict(D=128, P=64, lengths=[64, 1, 129], heads=(8, 8)),
        ):
            try:
                checked = run_case(mock=True, **case)
            except Exception:
                say(f"S0 self-check {case} EXCEPTION {traceback.format_exc()[-300:]}")
                continue
            say(
                f"S0 self-check {case} max_diff={checked['max_diff']:.3g} "
                f"out_sum={checked['out_sum']:.4g} (max_diff must be ~0)"
            )
    except Exception:
        say(
            "SECTION FAILED # S0: self-check of the probe's own reference against a plain-torch stand-in.",
            traceback.format_exc()[-800:],
        )

    try:
        # S1: main sweep.
        for D in (64, 128, 192, 224, 256):
            for P in (64, 128):
                for S in lengths_by_s:
                    for batch in (1, 4):
                        lengths = (
                            lengths_by_s[S]
                            if batch == 1
                            else [S, max(1, S // 2), S + 1, S]
                        )
                        for layout in ("contig", "scatter"):
                            guarded("S1", D=D, P=P, lengths=lengths, layout=layout)
    except Exception:
        say("SECTION FAILED # S1: main sweep.", traceback.format_exc()[-800:])

    try:
        # S2: number of blocks in the KV cache (the real pool is far larger than the minimal repro).
        for S in (2, 65, 300):
            for blocks in (1 if S == 2 else 2, 16, 256, 2048):
                guarded(
                    "S2",
                    D=256,
                    P=64,
                    lengths=[S],
                    layout="scatter",
                    num_blocks=max(blocks, (S + 63) // 64 + 2),
                )
    except Exception:
        say(
            "SECTION FAILED # S2: number of blocks in the KV cache (the real pool is far larger than the minimal repro).",
            traceback.format_exc()[-800:],
        )

    try:
        # S3: head configuration (TP2 gives 12/2 for Qwen3.6-27B).
        for heads in ((24, 4), (12, 2), (8, 8), (8, 1), (2, 1), (1, 1)):
            guarded("S3", D=256, P=64, lengths=[65], heads=heads)
    except Exception:
        say(
            "SECTION FAILED # S3: head configuration (TP2 gives 12/2 for Qwen3.6-27B).",
            traceback.format_exc()[-800:],
        )

    try:
        # S4: num_splits.
        for S in (2, 65, 2000):
            for splits in (1, None):
                guarded("S4", D=256, P=64, lengths=[S], num_splits=splits)
    except Exception:
        say("SECTION FAILED # S4: num_splits.", traceback.format_exc()[-800:])

    try:
        # S5: dtype.
        guarded("S5", D=256, P=64, lengths=[65], dtype=torch.float16)
        guarded("S5", D=128, P=64, lengths=[65], dtype=torch.float16)
    except Exception:
        say("SECTION FAILED # S5: dtype.", traceback.format_exc()[-800:])

    try:
        # S6: stability, fixed seed then varying seeds.
        for S in (2, 65, 2000):
            fixed = [run_case(D=256, P=64, lengths=[S]) for _ in range(20)]
            say(
                f"S6 fixed-seed D=256 P=64 S={S} repeats=20 distinct_outputs={len({r['digest'] for r in fixed})} "
                f"broken={sum(r['max_diff'] > TOL for r in fixed)} all_zero={sum(r['zero_rows'] == r['rows'] for r in fixed)} "
                f"out_sum_min={min(r['out_sum'] for r in fixed):.4g} out_sum_max={max(r['out_sum'] for r in fixed):.4g}",
            )
            varied = [
                run_case(D=256, P=64, lengths=[S], seed=100 + i) for i in range(20)
            ]
            say(
                f"S6 varied-seed D=256 P=64 S={S} repeats=20 broken={sum(r['max_diff'] > TOL for r in varied)} "
                f"all_zero={sum(r['zero_rows'] == r['rows'] for r in varied)} "
                f"out_sum_min={min(r['out_sum'] for r in varied):.4g} out_sum_max={max(r['out_sum'] for r in varied):.4g}",
            )
            for D in (64, 128):
                varied = [
                    run_case(D=D, P=64, lengths=[S], seed=100 + i) for i in range(20)
                ]
                say(
                    f"S6 varied-seed D={D} P=64 S={S} repeats=20 broken={sum(r['max_diff'] > TOL for r in varied)} "
                    f"max_diff_max={max(r['max_diff'] for r in varied):.4g}",
                )
    except Exception:
        say(
            "SECTION FAILED # S6: stability, fixed seed then varying seeds.",
            traceback.format_exc()[-800:],
        )

    if not MOCK:
        try:
            # S7: FlagSVT's minimal case verbatim in shape.
            torch.manual_seed(0)
            D, Hq, Hkv, S, P = 256, 24, 4, 2, 64
            k = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=DEV)
            v = torch.randn_like(k)
            kc = torch.zeros(1, P, Hkv, D, dtype=torch.bfloat16, device=DEV)
            vc = torch.zeros_like(kc)
            kc[0, :S] = k[0]
            vc[0, :S] = v[0]
            q = torch.randn(1, 1, Hq, D, dtype=torch.bfloat16, device=DEV)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=kc,
                v_cache=vc,
                cache_seqlens=torch.tensor([S], dtype=torch.int32, device=DEV),
                block_table=torch.zeros(1, 1, dtype=torch.int32, device=DEV),
                softmax_scale=D**-0.5,
                causal=True,
                softcap=0.0,
                num_splits=1,
            )
            out2 = flash_attn.flash_attn_func(q, k, v, causal=True)
            say(
                f"S7 FlagSVT-minimal paged out.abs().sum() = {out.float().abs().sum().item()} (bad -> 0.0)"
            )
            say(
                f"S7 FlagSVT-minimal non-paged out.abs().sum() = {out2.float().abs().sum().item()}"
            )
        except Exception:
            say(
                "SECTION FAILED # S7: FlagSVT's minimal case verbatim in shape.",
                traceback.format_exc()[-800:],
            )

    say("done")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        say("FATAL", traceback.format_exc())
