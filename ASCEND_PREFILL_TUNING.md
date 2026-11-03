# Ascend Qwen3.6 prefill tuning

These plugin patches target Qwen3.6-27B TP=2 BF16 prefill. All are **off by
default**. For **27B only**, set these before starting new workers:

```bash
export SGLANG_FL_FIA_TND_GQA=1
export SGLANG_FL_GDN_GATING_BLOCK24=1
export SGLANG_FL_PREFILL_NZ_GATE_UP=1
```

For **35B-A3B**, leave all three unset or `0`: its shapes do not benefit.
Keep the existing plugin/FlagGems configuration and ordinary decode graphs.
To roll back, unset these options and restart workers; no weight or source
restoration is needed. SGLang, FlagGems and `sgl_kernel_npu` are not modified.

## Scope and compatibility

Validated on Ascend 910C with image
`quay.io/ascend/sglang:v0.5.11-cann8.5.0-a3` and **installed** `sgl_kernel_npu`
2026.3.1, not its upstream checkout. Revalidate other software versions.
Measurements used TP=2, C=64, output 1024, prefill chunks 16384 and memory
fraction 0.82. MTP/speculation, prefix/radix cache and piecewise graphs were off.

| Option | Guarded shape | Change |
|---|---|---|
| FIA TND GQA | One fresh, unpadded 16384-token EXTEND, Q12/K2/V2/D256, BF16 | Use TND and return its output directly. Preserve causal mask/scale and the single KV write before attention. Accept contiguous V or zero-copy packed stride `(7168,256,1)`. |
| GDN gating BLOCK24 | Contiguous BF16 a/b `[16384,24]`, FP32 A_log, FP32/BF16 dt_bias | Reuse the installed kernel with one 24-head group instead of three groups of eight. Keep FP32 outputs, beta and threshold. |
| Ephemeral NZ gate-up | Contiguous BF16 x `[16384,5120]`, ND weight `[17408,5120]`, exact unquantized method, no bias | Convert the current weight to temporary FRACTAL_NZ for linear. Preserve the original ND Parameter/storage, with no persistent or stale-weight cache. |

Unverified shapes, dtypes, strides and attention modes fall back. Decode uses
the original implementations. NZ also falls back during graph capture because
CANN's ND-to-NZ Identity is not capturable; `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`
prevents its installation. Installation is idempotent and checks function/kernel
signatures, but cannot guarantee future compatibility from signatures alone.
Operator failures are not swallowed or retried after KV writes.

## Validation boundaries

FIA TND and BSND are **not bitwise equivalent**. The packed-V microtest measured
max absolute error 0.001953125 and relative RMS about 5.34e-5; near-tied logits
can change greedy tokens. Do not enable these options when exact baseline token
identity is required. Run task-specific quality evaluation: a small smoke/retrieval
corpus and sampled exact gating/GEMM tests do not prove model-level equivalence.
NZ validation includes in-place weight updates, ND storage and graph replay.

Use repeated same-device E2E tests across models/lengths, recording mean/median/p99
TTFT/TPOT and total/output throughput. Count unique implementation names in
`flaggems_ops.log`, frozen before extra quality/profiling requests. One-time
`FIA_TND_FAST_HIT`, `GATING24_FAST_HIT` and `PREFILL_NZ_FAST_HIT` messages confirm
dispatch, not operator counts; ranks and logging handlers may duplicate them.

## Opt-in 27B 16K/1K graph combination

For the measured 27B 16K/1K, TP2, C64 configuration only, the following
additional switches can be enabled together with the three 27B switches above:

```bash
export SGLANG_FL_FLA_SINGLE_SOLVE=1
export SGLANG_FL_FLA_HYBRID_CHAIN=1
export SGLANG_FL_NATIVE_CONV_SILU=1
export SGLANG_FL_CONV_COALESCED_WEIGHT=1
export SGLANG_FL_QWEN36_GRAPH_SYNC=1
export TRITON_ALL_BLOCKS_PARALLEL=1
```

Use page size 256 and a Triton cache **separate** from ABP-disabled runs;
`TRITON_ALL_BLOCKS_PARALLEL` is not reliably part of the cache key in this
runtime. Restart workers when changing these switches. For the other seven
model/length cases, leave the five new plugin switches off and use the original
page size 128 and ABP-disabled cache. The graph and convolution switches are
not inherently limited to 16K, so this deployment scope matters.

The FLA patch bypasses variable-length indexing only for a single fresh BF16
16K segment with the exact 27B GDN shapes; W/U retain the installed variable-
length kernels. The prefill convolution separates native SiLU from the CANN
convolution for sufficiently long inputs. Decode convolution reads four taps
per channel together from the original row-major weight, avoiding a captured
transpose. In text-only non-speculative decode, mRoPE positions come from the
NPU positions tensor, and graph replay uses SGLang's CPU length mirror. A
fresh completion event gates the next task-handle update before it can overlap
the preceding replay. Unsupported modes use the original methods.

On the pinned runtime, two clean-code 27B 16K/1K repeats measured 4113.394 and
4115.659 total tok/s versus a same-die 3942.511 baseline (+4.33%/+4.39%).
All seven 1K/4K/16K quality prompts matched a fresh page256 baseline for 128
output tokens each. Forcing these switches on two 64K prompts is outside the
measured deployment scope and changed their continuations; keep the new
switches off for 64K. With the production 64K settings (page128, ABP off,
new switches off), nine 27B quality outputs matched a same-configuration
baseline token for token. The target performance FlagGems inventory was 40
unique implementations versus 39 in the baseline, with no operator removed.
These are configuration-specific measurements, not a claim of general speedup
or bitwise equivalence outside the tested corpus. Enabling the new switches
on 35B changed two forced-continuation 64K outputs; keep them off for 35B
pending separate correctness work.
