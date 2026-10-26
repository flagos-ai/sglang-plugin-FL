# S5000 validation evidence and runtime boundary

**Current head: hardware revalidation pending.** Output completion now uses the
core Event path; eventfd was removed at `d4e22344de41e0b727243de15ab84bde68c40014`.
The later dispatch refactor also requires renewed eager/graph and model checks.
The September 20 hardware results below validate runtime commit
`f4bec310fd85e090ca994502881369d0222144b8`, which still used eventfd. They do
not validate the current head or establish its throughput.

The detailed experiment history, throughput tables and telemetry discussion
remain available in the [immutable September 20 record](https://github.com/CherryLemon/sglang-plugin-FL/blob/b3cd3b1d099d0a2e315680d43ab63f2cbae4a2ea/tests/musa/acceptance_20260920.md).
The repository keeps the correctness tools, dependency patch, source identities
and reproduction instructions needed for a fresh validation.

## Historical tested identity

- S5000 TP2, BF16 Qwen3.6-35B-A3B, SGLang 0.5.11 vendor image.
- Torch 2.9.0; imported TorchMUSA `2.9.0+ea1ca8d`;
  FlagTree `0.6.1+mthreads3.6` / imported Triton `3.6.0`.
- Original dependency arm: MATE `0.2.4.dev20260710`, TileLang
  `0.1.8+musa.3.git597eb92f`, TVM-FFI `0.1.9.post1+musa.1`.
- Upgraded arm: MATE `0.2.7`, TileLang `0.1.12+musa.2`,
  TVM-FFI `0.1.11.post1+musa.1`. SDK `4.3.5` and driver `3.3.5` stayed fixed.
- Vendor-core Python tree SHA-256:
  `1c9296f69f8c72867538c4a6cb74b90e60f8b333fdcd501dfa45c4ba0eec1fc4`.
- Exact historical environment: `qwen36_perf.env` at commit `b3cd3b1` above;
  the current environment file has eventfd removed and is not that identity.
- Pinned image/core/compiler dependencies remain required. The plugin wheel
  alone is not the historical validated service.

## Historical correctness evidence

| Gate | Recorded result at the historical runtime commit |
| --- | --- |
| Full target-image unit suite | 421 passed, 9 subtests passed |
| Real TopK/combine, eager and refreshed graphs | 6 passed; replay overwrites poisoned output |
| Fixed-work routed MoE | 17 shapes match reference hashes |
| Shared gate-tail | 11 cases, including BF16 materialization and NaN/Inf masks |
| M16K workspace | Reserved before KV sizing; pointer-stable reuse |
| Wheel contents | 7 native assets match source |
| TP2 startup | Both ranks capture all 12 decode buckets at `.970` |
| Patched model repeatability | Both stacks pass 17 cases and 4 semantic smoke requests |

Graph tests fork from the active capture stream, refresh inputs at fixed
addresses and poison outputs before replay. This corrected an earlier empty
capture that could pass by comparing an untouched eager output.

## C4 dependency defect and required patch

**Upgrading MATE alone does not fix C4 nondeterminism.** The original and
0.2.7 stacks both fail isolated GDN prefill repeatability without the H-ready
patch. First-token drift precedes decode graph replay; fixed KKT inputs isolate
the varying output/state to `fused_chunk_gdn_prefill`. The failure-sensitive
operation is reuse of the physical H-ready barrier across adjacent iterations;
precise silicon/compiler attribution remains unproven.

The [four-line dependency patch](mate-gdn-h-ready.patch) alternates two physical
H-ready barriers without changing math, state storage or H-free barriers.
The plugin does not apply it silently. On each patched dependency arm,
17 seeded cases × 100 cache-pressure repeats pass a CPU FP32 recurrence
reference, finite-output/state checks, exact repeatability and poisoned-output
coverage. Historical maximum synthetic errors were 0.000792258 for output
and 0.00635123 for final state on the original arm.

Verify `mate/gdn_kernels/tilelang/gdn_prefill.py` SHA-256 before and after patch:

| Source | Before | After |
| --- | --- | --- |
| Pinned 0.2.4 compatibility source | `b00245138dd001e1c5e4c0c584ef1d08555573c5fc424196b7a71f79c6ba0b2a` | `4848385c2703fe9877643045ce7a5e9fef7eea3c55e4cea04bf0da627e993c01` |
| Official 0.2.7 | `33253d1a163b8c3f0157fd568ea2d2a6881f9801b0fbc65d20faee201d0b0712` | `f9ed845705e4b183feeae55af0656492abc57f7ed66af69d5655321a5d0c6f01` |

Use [the seeded reproducer](repro_mate_gdn_prefill.py) and
[validation handoff](README.md) with the exact delivered dependency source.

## Limits and current acceptance gates

The two patched stacks are individually repeatable but their full-model
token/logprob outputs are not bitwise equivalent. Bounded semantic smoke does
not establish dataset accuracy. The old nonstreaming benchmark does not
establish TTFT/TPOT or an SLA; its small short-input gains and slight 64K
regression measure MATE together with TileLang/FFI, not MATE alone.

For the current head, rerun installation/asset checks, actual dispatch-policy
hits, eager/refreshed graphs, two-stream combine, shared BF16 boundaries,
M16K scratch reuse and TP2 C1/C4 token/logprob repeatability on the pinned image.
Record the plugin SHA, image/core/model identities, patched MATE hash and logs.
Collect any new performance result only after these correctness gates pass.
