# Qwen3.6 MUSA plugin-side integration

Status: fixed-work GPU relocation and replay-refresh gates pass on the pinned
image. **Full-service C4 generated-token parity has not passed:** repeated
requests also diverge within the retained reference service, so this is not
an established migration-specific regression, nor evidence of equivalence.
Final-image performance acceptance is separate. Historical service throughput
must not be attributed to this source layout before that validation.

## Baseline and ownership

The reference is the SGLang 0.5.11 MUSA vendor image, with core commit
`612785ffdcaf35552f1ed433a981d596ca9fe900` **plus its existing vendor changes**.
This is not an unmodified upstream SGLang checkout. The campaign added nine
core files on top; this candidate moves those additions into the plugin:

| Former core change | Plugin owner |
| --- | --- |
| `environ.py` capability flag | `moe/dispatch.py`, reads the existing opt-in environment variable |
| MUSA FA3 capture metadata | `patches/fa3_graph_metadata.py` |
| Triton runner quant-info field | Existing MThreads fused-MoE dispatch, no dataclass ABI change |
| MoE sequence, M4 scheduling, M16K scratch | `moe/fused_moe.py` |
| Fused SwiGLU kernel and launcher | `moe/kernels.py` |
| Shared expert gate-tail kernel | `moe/shared_expert_gate_tail.py` |
| Unquantized post-load capability | Loaded-weight guards in `moe/dispatch.py`, no weight mutation |
| ModelRunner pre-KV allocation | `patches/moe_workspace.py` |
| Qwen shared-expert call site | `patches/shared_expert_gate_tail.py` |

No installed SGLang file is rewritten at startup; no copied `sglang` package
is put ahead of the original on `PYTHONPATH`. Existing vendor-image core
changes remain a pinned dependency, not a new plugin-only claim.

Custom-op names are distinct from SGLang's registrations. Scheduling, route
alignment and reduction are resolved through the original module at call time
so the existing plugin patches (especially deterministic combine) remain
effective.

## Supported candidate contract

Qwen3.6-35B-A3B BF16, MTT S5000, TP2, local 256 experts, top-k 8, canonical
W13 `(256, 512, 2048)` and W2 `(256, 2048, 256)`, no EP dispatch, LoRA, EPLB,
router-input weighting, quantization, bias or fused MoE all-reduce. The
release path uses eager prefill and full decode graphs; piecewise graphs are
not admitted by the adapter. Unsupported calls keep `forward_musa` unchanged.

The routed adapter rechecks loaded weights rather than caching a capability
in a core dataclass. A launch/allocation error is propagated, never retried
against possibly modified in-place inputs.

Retained opt-in controls:

- `SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE=1`: canonical-W13 fused prefill;
  the kernel sequence still disables fusion below 2048 tokens.
- `SGLANG_MUSA_M4_W13_BN64=1`: exact M4 up tile; original down/alignment config
  is not modified. The reference also enables the R2 M4 baseline resolver.
- `SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE=1`: reserve the 512 MiB
  fixed M16K routed-down scratch before KV-memory sizing, under the exact
  model/TP2/BF16 guard. The hook runs just before `init_memory_pool`; relative
  ordering against deterministic-mode setup is different from the former
  inline call and requires checking if that mode is enabled.
- `SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED=1`: narrow text-only M4 decode
  gate-tail path, with the original router, in-place addition and TP collective.
- `SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY=0`: diagnostic opt-out of the paged
  graph capacity correction. It is on by default for the supported inherited
  MUSA capture API. Existing backend overrides and changed signatures are
  left untouched.

The compiler contract is FlagTree 0.6.1+mthreads3.6 / Triton 3.6.0, not vendor
Triton 3.2. No T3.2 compatibility overlay is part of this release candidate.
The service also depends on the separately pinned MATE compatibility package;
moving the MoE code does not replace or remove that dependency.

Validated base image (not the final release image):

```text
harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-bash-mthreads-tree_0.6.1_mthreads3.6-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.10.12-torch_2.9.0-pcp_musa4.3.5-driver_3.3.5_server:202608182028
Repo digest: sha256:9b9c082f9af577de9156414869ce93ed3a06dedf7bcf63e0dfed6be14560a339
Image ID: sha256:871ac919ba253a0d750f52d613804963133612d63c7c8db139ef2b2c46884ae3
```

The tested interpreter provides TorchMUSA `2.9.0+ea1ca8d`; the image's system
Python is not interchangeable with that environment. The MATE compatibility
source mapping digest is
`3e9670b579dd911d0967bfe07bd762e99554da35bfb5b11ab299016682de1387`.
The immutable final image must carry both the plugin and that dependency;
installing this wheel alone into an arbitrary runtime is not the validated
service contract.

## Patch background and equivalence notes

These notes record experiment, measurement and equivalence context that
previously lived in the patch source. They describe the snapshot
`b6995b4d72df142291fe00537055a55ab9f1a3bd`, not the refactored source, and
must not be read as post-refactor validation or as a new performance claim.

- **Softmax TopK schedule.** The MUSA kernel carries fifteen launch
  configurations, and autotuning them also flushes a 256 MiB cache buffer
  between candidates, making the first large prefill wave several seconds
  slower on MP31. `warps=1, stages=1` is the measured choice for the
  `E=256, K=8` graph and 1K-16K prefill shapes. The pinned path calls the
  inner `JITFunction.run` directly instead of mutating `kernel.configs`; for a
  single-config autotuner branch this forwards the same arguments to the same
  `fn.run`, and the skipped shared-state writes have no readers outside the
  tuner. `JITFunction.run` takes the caller current stream and `do_bench`
  takes no stream parameter, so the pinned call inherits the caller stream;
  the raw fallback path keeps the stock autotuner and its first-key host-state
  race is not fixed.
- **MoE schedule.** TorchAda's bundled Triton 3.2 uses eight warps and a K=128
  tile for the Qwen3.6-35B-A3B TP2 decode shape, which has a large performance
  cliff on some S5000 systems with Triton 3.6. A four-warp, K=64 configuration
  is within a few percent of the old-system optimum and about three times
  faster on the affected systems. Long-prefill profiling found the generic
  M=64/N=64/K=32 tile left expert-padding and occupancy performance unused;
  the measured M=32/N=128/K=64, eight-warp, one-stage schedule is 12-27%
  faster across 8192-token random, balanced-shuffled and block-boundary
  routes, and across 2048/4096/6144/8192-token chunks. The M=16384 schedule is
  a separate fixed-work confirmation at M=64/N=128/K=64; no intermediate or
  adjacent token count is widened.
- **Deterministic combine.** The standalone screen showed the three-kernel
  combine chain can be replaced by one ordered-FP32, no-atomic Triton kernel
  for the S5000 Qwen3.6 TP2 contract. For decode-graph M=40/M=64, the opt-in
  keeps Qwen's two-stream overlap: the shared branch stays on the primary
  stream, routed experts stay on `alt_stream`, and the tail wait at the reduce
  seam joins them immediately before the fused consumer.

## Required validation before release

For the current checkout, use the [S5000 validation handoff](../../../../../../tests/musa/README.md)
for CPU regression checks, real TopK/combine refresh tests, the TP2 full-graph
smoke case and separate C1/C4 repeatability records. These are validation tools;
adding them does not change the historical acceptance status below.

Validated code snapshot: `b6995b4d72df142291fe00537055a55ab9f1a3bd`.

- Local full platform suite: 200 passed, one pinned-image API check skipped;
  nine subtests passed. On the pinned image the MUSA subset passes all 189
  tests, including that API check, plus nine subtests.
- All 12 production decode graph buckets (1, 2, 4, 8, 12, 16, 24, 32, 40,
  48, 56, 64) and eager M2047/2048/2049/8192/16384: fixed synthetic A/W/routes
  produce bitwise-identical BF16 outputs against the captured core-overlay
  reference. Each decode graph refreshes A, expert IDs and routing weights
  through fixed-address buffers (phase 0/1/0); all outputs match eager and the
  other arm. Ten deterministic repeats per case pass.
- The 512 MiB M16K scratch is pointer-stable and actually reused.
- Shared gate-tail: 11 input cases, including zero, NaN and Inf, pass the
  materialized MUSA oracle at logits/probability/output BF16 boundaries,
  graph refresh, ten repeats and output sentinels. NaN/Inf masks are checked
  separately from finite bitwise values. Both source layouts agree exactly.
- Installed-wheel hook/import checks pass; all eight existing baseline core
  files retain their hashes, and the extra shared helper is absent from core.

These are source-relocation correctness results, not a formal model accuracy
score, full-service route-equivalence claim or sanitizer/performance result.

Full-model startup with the installed plugin and baseline image core passes
at mem-fraction 0.970: both ranks reserve scratch before KV sizing and capture
all 12 graph buckets. Initial C1 1024/32 requests match the retained service's
generated tokens and output logprobs exactly. C4 1024/32 does not meet the
generated-token parity or repeated-call determinism gate. Three subsequent
interleaved runs per service reproduce reference self-drift as well as
candidate self-drift; request IDs confirm output ordering. First-token
logprobs already vary, so the mismatch cannot be assigned solely to the
M4 decode gate-tail. The reference uses 0.965 on another card pair; these
are functional checks, not a single-variable full-model causal experiment.
Keep the failed records and leave formal accuracy/release acceptance open.

Sanitized comparison records are identified by SHA256:

```text
initial MoE cases: a56bcdf808be54abcd18eb8967085c5af73b6c9ebcd64b354ed1004bdf0a0ed5
remaining buckets: 286d35439bc68f1eefc4be9c12ff1f5313cb5c3d2e797cdabdfdc36fd60f4060
shared gate: 8571424fdd2aa37748b2eac4051b7fbf1aa241988e31ac8511695bc6cb28b466
```

Run the platform unit tests; then on the pinned image verify installation and
actual hook/dispatch hits, immutable baseline core hashes, loaded-weight and
route parity, eager/full-graph outputs, BF16 materialization boundaries, M4
shared-expert reduction ownership, and M16K workspace allocation before KV
sizing. Recapture source-location-sensitive compiled artifacts and benchmark
the candidate itself. The existing engineering matrix is not the formal
FlagRelease performance/accuracy acceptance matrix.

Operator handoff: FlagGems owns the fused SwiGLU and gate-tail device kernels;
FlagTree owns scheduling/fusion selection and graph/dispatch integration.
The unchanged upstream fallback remains the compatibility boundary.
