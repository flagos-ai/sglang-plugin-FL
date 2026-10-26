"""Independent seeded GDN recurrence correctness and repeatability probe."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path

import tilelang
import torch
import torch_musa
from mate.gdn_kernels.tilelang.gdn_kkt_solve import kkt_solve


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def reference(q, k, v, g, beta, cu, state):
    result = torch.empty_like(v, dtype=torch.float32)
    final = state.clone()
    for seq, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        h = state[seq].clone()
        for token in range(int(start), int(end)):
            kt = k[0, token].float().repeat_interleave(2, dim=0)
            qt = q[0, token].float().repeat_interleave(2, dim=0)
            h.mul_(g[0, token].exp()[:, None, None])
            residual = v[0, token].float() - (h @ kt[..., None]).squeeze(-1)
            residual.mul_(beta[0, token, :, None])
            h.add_(residual[..., None] * kt[:, None, :])
            result[0, token] = (h @ qt[..., None]).squeeze(-1) * (128**-0.5)
        final[seq] = h
    return result, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-file",
        type=Path,
        help="Optional standalone candidate; default: installed MATE",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    assert not args.output.exists()
    if args.kernel_file is None:
        from mate.gdn_kernels.tilelang import gdn_prefill as mod

        args.kernel_file = Path(inspect.getfile(mod))
    else:
        spec = importlib.util.spec_from_file_location("candidate_gdn", args.kernel_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(args.seed)
    cases = (
        [(4, 1024, "contiguous", False)]
        if args.smoke
        else [
            (4, n, "contiguous", False)
            for n in (1, 63, 64, 65, 127, 128, 129, 257, 1024, 1025, 2048)
        ]
        + [
            (1, 16384, "contiguous", True),
            (4, 4096, "contiguous", False),
            (4, 1024, "split_qkv", False),
            (4, 1024, "contiguous", True),
            (4, 1025, "split_qkv", True),
            (4, (1, 63, 129, 1025), "split_qkv", True),
        ]
    )
    record = {
        "status": "running",
        "seed": args.seed,
        "cases": [],
        "kernel_sha256": hashlib.sha256(args.kernel_file.read_bytes()).hexdigest(),
        "cache_pressure_bytes": 128 * 1024 * 1024,
        "performance_measured": False,
        "reference": "CPU FP32 scalar-token GDN recurrence, BF16 normalized Q/K",
        "output_tolerance": {"atol": 0.02, "rtol": 0.02},
        "state_tolerance": {"atol": 0.04, "rtol": 0.02},
    }
    record["gpu"] = torch.musa.get_device_name(0)
    record["versions"] = {
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "mate": importlib.metadata.version("mate"),
        "tilelang": tilelang.__version__,
        "apache-tvm-ffi": importlib.metadata.version("apache-tvm-ffi"),
    }
    record["import_origins"] = {
        "kernel": str(args.kernel_file.resolve()),
        "kkt": inspect.getfile(kkt_solve),
        "tilelang": tilelang.__file__,
        "torch": torch.__file__,
        "torch_musa": torch_musa.__file__,
    }
    assert "S5000" in record["gpu"], "This reproducer is scoped to the S5000 campaign"
    evict = torch.empty(record["cache_pressure_bytes"] // 4, device="musa")
    try:
        for batch, length, layout, nonzero in cases:
            lengths = [length] * batch if isinstance(length, int) else list(length)
            record["active_case"] = {
                "lengths": lengths,
                "layout": layout,
                "nonzero_initial_state": nonzero,
            }
            total = sum(lengths)
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32
            )
            q, k = [
                torch.nn.functional.normalize(
                    torch.randn(1, total, 8, 128, generator=generator), dim=-1
                ).to(torch.bfloat16)
                for _ in range(2)
            ]
            v = torch.randn(1, total, 16, 128, generator=generator).to(torch.bfloat16)
            g = -0.005 - torch.rand(1, total, 16, generator=generator) * 0.1
            beta = torch.randn(1, total, 16, generator=generator).sigmoid()
            initial = (
                torch.randn(batch, 16, 128, 128, generator=generator) * 0.125
                if nonzero
                else torch.zeros(batch, 16, 128, 128)
            )
            expected, expected_state = reference(q, k, v, g, beta, cu, initial)
            if layout == "contiguous":
                qr, kr, vr = [x.musa() for x in (q, k, v)]
            else:
                mixed = torch.empty(
                    total, 32 * 128, dtype=torch.bfloat16, device="musa"
                )
                qr = mixed[:, :1024].view_as(q)
                kr = mixed[:, 1024:2048].view_as(k)
                vr = mixed[:, 2048:].view_as(v)
                for dst, src in zip((qr, kr, vr), (q, k, v)):
                    dst.copy_(src)
            gr, br, cr, hr = [x.musa() for x in (g, beta, cu, initial)]
            a = kkt_solve(k=kr, b=br, cu_seqlens=cr)
            output = torch.empty_like(vr)
            final_state = torch.empty_like(hr)
            inputs = [qr, kr, vr, a, gr, br, cr, hr]
            before = [digest(x) for x in inputs]
            first = first_state = None
            hashes, state_hashes = set(), set()
            for repeat in range(args.repeats):
                output.fill_(float("nan"))
                final_state.fill_(float("nan"))
                evict.fill_(repeat + 1)
                result, _, state = mod.fused_chunk_gdn_prefill(
                    q=qr,
                    k=kr,
                    v=vr,
                    a=a,
                    g=gr,
                    b=br,
                    initial_state=hr,
                    output_final_state=True,
                    cu_seqlens=cr,
                    scale=128**-0.5,
                    chunk_size=64,
                    is_log_space=True,
                    output=output,
                    output_state=final_state,
                )
                result, state = result.cpu(), state.cpu()
                assert torch.isfinite(result).all() and torch.isfinite(state).all()
                hashes.add(digest(result))
                state_hashes.add(digest(state))
                if first is None:
                    first, first_state = result.clone(), state.clone()
                    torch.testing.assert_close(
                        result.float(), expected, atol=0.02, rtol=0.02
                    )
                    torch.testing.assert_close(
                        state, expected_state, atol=0.04, rtol=0.02
                    )
                assert torch.equal(result, first), f"output drift at repeat {repeat}"
                assert torch.equal(state, first_state), (
                    f"state drift at repeat {repeat}"
                )
            assert [digest(x) for x in inputs] == before, "input mutation"
            err = (first.float() - expected).abs()
            herr = (first_state - expected_state).abs()
            row = {
                "lengths": lengths,
                "layout": layout,
                "nonzero_initial_state": nonzero,
                "repeats": args.repeats,
                "unique_outputs": len(hashes),
                "unique_states": len(state_hashes),
                "max_abs_error": float(err.max()),
                "rms_error": float(err.square().mean().sqrt()),
                "state_max_abs_error": float(herr.max()),
                "state_rms_error": float(herr.square().mean().sqrt()),
            }
            record["cases"].append(row)
            args.output.write_text(json.dumps(record, indent=2))
            print(json.dumps(row), flush=True)
        record.pop("active_case", None)
        record["status"] = "pass"
    except BaseException as exc:
        record["status"] = "fail"
        record["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
