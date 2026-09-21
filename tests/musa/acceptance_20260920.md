# S5000 release/perf integration acceptance — 2026-09-20

Status: **hardware correctness gates and historical performance reproduction passed**
for the recorded original MATE stack with the explicit H-ready dependency
patch. The paired dependency comparison is complete. The upgrade has small
short-input gains and a slight 64K regression; cross-version dataset accuracy
has not been established. The runtime dependency requirements below are part
of this acceptance.

This is a historical snapshot of the commits listed below, which included
eventfd completion. The current code and performance profile remove eventfd;
these measurements do not validate that subsequent change.

## Code and runtime

- Release parent: `690168f0f39e0a4ea0284c23b4bd95326aef6d8f`.
- Perf parent: `f59269f8166512eeb8f3800f6ed69e518c6661d2`.
- Integration: `b2453db9010da222d816ed984c899f1643ce4ecb`.
- Replay-test correction: `a4a8959051813a463dece4ada6df94f843abda64`.
- FMHA API compatibility: `f4bec310fd85e090ca994502881369d0222144b8`.
- S5000 TP2, Torch 2.9.0, TorchMUSA 2.9.0+ea1ca8d, SGLang 0.5.11,
  FlagTree 0.6.1+mthreads3.6 / Triton 3.6.0, MATE 0.2.4.dev20260710.
- Original paired dependencies: TileLang `0.1.8+musa.3.git597eb92f` and
  TVM-FFI `0.1.9.post1+musa.1`.

The release-side module layout, scheduling guards, FLA loader and custom-AR
workspace sizing are retained. Perf contributes eventfd completion, native
copy/index-put defaults and FlagCX self-copy avoidance. Existing release tests
supersede older duplicate perf tests; additional behavioral tests live under
`tests/unit_tests/platform/musa`.

The pinned vendor SGLang core was unchanged for acceptance. Its Python tree digest was
`1c9296f69f8c72867538c4a6cb74b90e60f8b333fdcd501dfa45c4ba0eec1fc4`.
The exact historical environment is captured by `qwen36_perf.env` at commit
`b3cd3b1d099d0a2e315680d43ab63f2cbae4a2ea`; explicit
environment blacklists override YAML, including the native `index` selection.

## Completed gates

| Gate | Result |
| --- | --- |
| Target-image full unit suite after FMHA compatibility fix | 421 passed, 9 subtests passed |
| Real TopK / combine, eager and refreshed graph replay | 6 passed |
| MoE fixed input / weights / routes | 17 shapes match reference output hashes |
| M16K workspace | Allocated before KV sizing, reused without pointer changes |
| Shared expert gate-tail | 11 cases pass, including materialization boundaries and NaN/Inf masks |
| Native wheel assets | All 7 native files present and byte-identical to source |
| `.970` TP2 startup | Both ranks capture all 12 decode buckets; no OOM |
| Short C1/C4 requests at `.970` | 24 requests, exact token/logprob self-repeatability |

The original GPU test exposed a native CUDA stream call on MUSA. That call was
corrected. A subsequent empty-graph warning exposed a test defect: it forked
from the pre-capture stream and compared against an eager result still in the
output buffer. The test now forks from the active capture stream and poisons
outputs before every replay. All six cases pass with no empty-graph warning.

## Initial model comparison at `.965` before the MATE fix

Both services used the historical optimized settings and direct token IDs.
Greedy requests had fixed completion lengths and returned native token IDs and
logprobs. The retained reference remained unchanged.

| Concurrent requests | Input / output | Token and logprob parity | Candidate token repeats |
| ---: | ---: | --- | --- |
| 1 | 1024 / 32 | pass | pass |
| 4 | 1024 / 32 | **fail** | **fail** |
| 1 | 4096 / 32 | pass | pass |
| 64 | 64 / 8 | pass | pass |
| 1 | 16384 / 8 | pass | pass |

Three further interleaved rounds checked response IDs and ordering. The
reference changed 27 tokens across two prompts in one repeat comparison. The
candidate changed 61 tokens across two prompts in another. Request ordering
matched throughout. Logprobs also drifted when generated token IDs happened to
remain equal. This reproduces the historical open gate; it neither establishes
a migration-specific regression nor proves full-model equivalence.

## C4 diagnosis

The failure also occurs with one output token, before decode graph replay.
First-token logprobs vary for C4 input lengths 256, 1023, 1024 and 1025, and
C2 input length 2048. This is not confined to a 4096-token prefill boundary.

A separate diagnostic container instrumented prefill only. Across four repeated
C4 requests on both TP workers, layer-0 hidden inputs, projections, Q/K/V, gates,
sequence offsets and selected zero initial states were identical. The first
different tensor was the output of the MATE GDN kernel. Later MoE inputs were
already different. Instrumentation remained private and was not added to the
release plugin or the accepted core source contract.

Real inputs captured from both workers reproduce the failure in a standalone
MATE process without serving or FlagGems operator registration. Every iteration
reloads Q/K/V and gates and resets state; this matters because MATE normalizes
Q/K in place. Both contiguous and split-QKV layouts can fail.

Splitting the operation further isolates the failing execution path:

- The KKT solve is repeatable for the captured case.
- With normalized Q/K and the KKT result held fixed, repeated calls to
  `fused_chunk_gdn_prefill` produce different outputs and final states.
- This localizes the problem to that kernel's execution path. It does not yet
  establish whether the underlying defect is kernel synchronization, compiler
  lowering, or another runtime issue.

FlagGems PR #6492 was tested in its own container. Its patched generic MoE
function was imported but was not called by this configuration; the plugin-owned
MoE path was active. C4 still failed. The patch was not adopted.

## Official MATE upgrade experiment

A fresh container tested MATE **0.2.7**, TileLang-MUSA **0.1.12+musa.2** and
apache-tvm-ffi **0.1.11.post1+musa.1**, using official MUSA wheels verified by
SHA-256. Import origins were checked. Dependencies were installed into an
isolated target directory with a fresh compilation cache. Torch/TorchMUSA,
SDK **4.3.5** and driver **3.3.5** stayed fixed. No host upgrade was performed.

Each input/layout case below used 100 repetitions on an S5000:

| Captured TP worker | QKV layout | Distinct output hashes | Maximum repeat difference |
| --- | --- | ---: | ---: |
| 0 | contiguous | 36 | 0.56640625 |
| 0 | split-QKV | 19 | 0.56640625 |
| 1 | contiguous | 33 | 0.6484375 |
| 1 | split-QKV | 5 | 0.53125 |

No NaN or Inf appeared. In a further 100-repetition stage test, KKT produced one
output hash with zero difference; the fixed-input recurrence produced five
hashes with maximum difference **0.6484375**. Thus this paired dependency
upgrade does **not** resolve the observed C4 failure on the current SDK/driver.
An SDK/compiler upgrade has not been tested. This unpatched dependency arm was
excluded from full-model performance because the isolated correctness gate failed.

A separate diagnostic candidate adding a 128-thread barrier before the
recurrence's `Ag @ W` multiply also failed repeatability. It was not adopted.
The upgrade alone is not a C4 fix.

## H-ready barrier diagnosis and candidate fix

Holding normalized Q/K, KKT output, all other inputs and device addresses fixed,
then filling a separate 128 MiB device buffer before each invocation, makes the
original recurrence fail reliably. This creates cache pressure; no measured
cache-eviction or cache-hit counter is claimed.

The two compute consumers of `h_shared` can enter waits for adjacent iterations
on the same physical `h_is_ready` barrier. Replacing it with two alternating
physical barriers, each using phase `(iteration // 2) % 2`, removes the observed
failure. The state buffer, `h_is_free`, math and compute overlap are unchanged.
The patch adds one physical barrier and changes four source lines.

| Controlled fixed-input experiment | Repeats | Distinct outputs / states |
| --- | ---: | ---: |
| Original MATE 0.2.4 before candidate | 100 | 100 / 100 |
| Two H-ready barriers, original dependency stack | 100 | 1 / 1 |
| Original MATE 0.2.4 after candidate | 100 | 100 / 100 |
| Two H-ready barriers, official 0.2.7 dependency stack | 100 | 1 / 1 |
| Four / sixteen H-ready barriers, official stack | 100 each | 1 / 1 each |

Rotating only the V-ready, V-free, V-delta-ready or H-free barriers does not
resolve the failure. Waiting for the prior output iteration, or adding a
256-thread compute rendezvous, also resolves it but constrains overlap more
than the H-ready change.

Output-side instrumentation preserves the original failure: `QH` and `Pg` are
identical across 50 invocations, while the observed `Vdelta` changes. Its first
observed difference is at sequence offset 128. Instrumentation inside other
compute groups can suppress the failure and is not used to declare the
uninstrumented kernel correct.

This identifies H-ready barrier reuse as the failure-sensitive operation. It is
consistent with the PH1 cross-phase waiter behavior documented in another
[upstream MP31 kernel](https://github.com/MooreThreads/mate/blob/v0.2.7/include/mate/attention/msa/collective/mp31_msa_maxscore_collective_tme_warpspecialized.hpp#L145-L150).
The exact hardware/compiler mechanism has not been independently proven.

A second probe attaches an iteration number to each H publication and records
it immediately after the consumers' H-ready waits. All 50 original-kernel runs
observe stale generations, for example consumer 0 in iteration 2 reads
generation 1. All 50 two-barrier runs observe the expected generation and
identical outputs. This demonstrates a failure of wait/publication ordering in
the instrumented original kernel; attribution to a specific silicon/compiler
defect still requires upstream confirmation.

The minimal patch also passes 17 seeded input cases, 100 cache-pressure repeats
each, on the original dependency stack. Coverage includes lengths 1, 63, 64, 65,
127, 128, 129, 257, 1024, 1025, 2048, 4096 and 16384; contiguous and split-QKV
layouts; unequal sequence lengths; zero and nonzero initial states. Every
output and final state is finite and bitwise repeatable. Inputs are unchanged,
and poisoned output buffers are fully overwritten. Against a literal CPU FP32
recurrence, the largest output error is 0.000792258 and state error 0.00635123.
These synthetic-input errors are separate from the captured model-input case.

The earlier 256-thread workaround was tested in a fresh full-model service
using the original MATE 0.2.4 stack, memory fraction `.970` and decode graphs.
Six input/concurrency pairs (C4/I1024, C1/I4096, C4/I256, C4/I1023, C4/I1025,
C2/I2048) each pass eight first-token repeats and twenty 32-token repeats, with
exact token/logprob equality. C64/I64, C4/I4096, C1/I16384, C4/I16384 and
C1/I65536 each pass five 32-token repeats. The smaller H-ready patch subsequently
passes the same complete matrix: all 17 case-level token/logprob records match
the workaround exactly. Both TP ranks capture all 12 decode buckets at `.970`
and confirm the patched MATE source. Four concurrent semantic smoke requests
also pass: integer addition, Chinese translation, sorting and JSON generation.
This is a bounded smoke check, not formal dataset accuracy or performance timing.

The dependency fix is retained as [mate-gdn-h-ready.patch](mate-gdn-h-ready.patch),
with an independent [seeded reproducer](repro_mate_gdn_prefill.py). Verify these
SHA-256 identities for `mate/gdn_kernels/tilelang/gdn_prefill.py`:

| Source | Before patch | After patch |
| --- | --- | --- |
| Pinned 0.2.4 compatibility source | `b00245138dd001e1c5e4c0c584ef1d08555573c5fc424196b7a71f79c6ba0b2a` | `4848385c2703fe9877643045ce7a5e9fef7eea3c55e4cea04bf0da627e993c01` |
| Official 0.2.7 | `33253d1a163b8c3f0157fd568ea2d2a6881f9801b0fbc65d20faee201d0b0712` | `f9ed845705e4b183feeae55af0656492abc57f7ed66af69d5655321a5d0c6f01` |

Both patched source identities match the tested standalone candidates.
This changes the MATE dependency contract; it is not a plugin-only comparison.

## MATE 0.2.7 serving compatibility

The first upgraded full-model startup failed before completing graph capture:
the plugin's FMHA selector wrapper accepted eight arguments, while MATE 0.2.7
passes a ninth `is_high_regpressure` argument. This was an API failure, not an
OOM; no memory fraction was lowered and no throughput was recorded for it.

The wrapper now forwards an explicitly supplied register-pressure argument and
leaves omitted arguments to the original selector. Thus the old eight-argument
contract is preserved. High-pressure cases retain MATE's own configuration;
the measured 8K pack-GQA optimization remains active for ordinary cases.
Eight focused tests pass, and calls to the actual 0.2.4 and 0.2.7 selectors
confirm forwarding, all aliases, idempotence and retained 8K tuning. The target
full unit suite passes 421 tests and 9 subtests. All six real GPU eager/graph
tests also pass with the upgraded dependencies; all seven native wheel assets
match the committed source. A fresh upgraded model service captures all 12 graph
buckets on both ranks at `.970`, passes the same 17 model-repeatability cases,
and passes all four semantic smoke requests.

The two patched dependency versions are internally repeatable but not bitwise
equivalent to each other. The six first-token cases choose the same tokens;
their largest selected-token logprob difference is 0.0654343. Five multi-token
cases diverge in token choice, including C64/I64 and C1/I65536. The remaining
cases can also differ in logprobs. These differences are retained as a separate
upgrade-accuracy question; repeatability and bounded semantic smoke do not
establish full dataset accuracy or cross-version equivalence.

A bounded decode reference check covers batches 1, 4 and 64, eight recurrent
steps and three repetitions with cache pressure. Both dependency versions pass
CPU FP32 comparison and produce identical BF16 output hashes. FP32 final-state
hashes differ between versions but repeat exactly within each version. Maximum
absolute errors are 0.000236542 for BF16 output and 2.98024e-7 for FP32 state.
This check did not reproduce a decode-state defect; it does not localize the
remaining model-level cross-version numerical differences. The standalone
diagnostic workload finished before formal timing and was stopped with release
evidence preserved.

Both throughput arms used this same plugin commit. Runtime imports still
select TorchMUSA `2.9.0+ea1ca8d` and Triton `3.6.0`; stale distribution metadata
for Triton reports `3.1.0`, so imported module versions/origins are recorded
separately rather than treating that metadata as an active-runtime change.

Relevant upstream references: [MATE GDN contract](https://mate-docs.mthreads.com/latest/gdn.html),
[MATE installation requirements](https://mate-docs.mthreads.com/latest/install.html),
[MATE v0.2.7 release](https://github.com/MooreThreads/mate/releases/tag/v0.2.7),
and [FlagGems PR #6492](https://github.com/flagos-ai/FlagGems/pull/6492).

## Matched end-to-end performance

Both arms use plugin runtime commit `f4bec310fd85e090ca994502881369d0222144b8`,
the same image, vendor core, physical S5000 TP2 pair, BF16 model, environment
profile and performance arguments. The original arm ran first, followed by the
upgraded arm. Both use the exact uninstrumented H-ready patch. The changed
runtime components are MATE plus its required TileLang/TVM-FFI combination;
these measurements do not isolate MATE alone.

Both stacks first boot and capture all 12 decode buckets on both ranks at
memory fraction `.970`. The separate `.965` comparison matches the retained
September-17 contract; this was not an OOM fallback. Both `.965` services
allocate 4,190,144 KV tokens, reproduce all 17 of their own `.970` token/logprob
records exactly and pass all four semantic smoke requests before timing.

Each shape uses one excluded full warmup and five measured rounds, N256/C64,
output length 1024, seed 0, greedy sampling with ignore-EOS, nonstreaming direct
integer token IDs. Timing covers client request submission through complete
response collection, including prefill and decode. Tokenizer loading, workload
construction and warmup are excluded. All 40 measured rounds complete 256
requests with the expected input/output usage, HTTP status and finish reason.
Prompt digests match the historical workload and both dependency arms.

Values below are five-round median **end-to-end output tokens/s**:

| Input / output | Historical | Patched original MATE | Patched MATE 0.2.7 | Upgrade change |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 1024 | 1744.06 | 1751.58 | 1784.41 | +1.87% |
| 4096 / 1024 | 1390.23 | 1388.79 | 1403.13 | +1.03% |
| 16384 / 1024 | 727.92 | 725.43 | 726.48 | +0.15% |
| 65536 / 1024 | 224.10 | 223.77 | 222.22 | -0.69% |

The original patched stack reproduces all four historical medians within 0.5%.
The dependency upgrade gives small improvements at 1K and 4K, is nearly flat at
16K, and changes 64K throughput by **-0.69%**. It does not provide a
uniform end-to-end improvement across these workloads. Cross-version numerical
equivalence and dataset accuracy remain unestablished, as detailed above; the
validated original stack remains the acceptance baseline.

Observed five-round ranges and population CVs:

| Input tokens | Original range / CV | Upgraded range / CV |
| ---: | ---: | ---: |
| 1024 | 1742.86–1758.18 / 0.329% | 1765.64–1787.81 / 0.462% |
| 4096 | 1388.45–1389.12 / 0.016% | 1403.06–1403.46 / 0.010% |
| 16384 | 725.37–725.62 / 0.012% | 726.33–726.60 / 0.013% |
| 65536 | 223.70–223.86 / 0.026% | 222.04–222.41 / 0.056% |

No thermal slowdown is recorded on either card during the measured client
command intervals. Graphics clocks were dynamic under unchanged device
settings; no fixed-clock or hardware-peak claim is made. There was no concurrent
diagnostic GPU workload during formal timing. Telemetry summaries cover client
command intervals, which also include tokenizer startup; throughput itself uses
the narrower request-only interval defined above.

No latency SLA was specified. This nonstreaming test does not measure TTFT or
TPOT. No formal dataset accuracy, sanitizer result or new profiler counter is
claimed. Raw logs, source identities, model output, telemetry and ownership
manifests remain in the private session artifacts.
