# MThreads Qwen3.6 MoE integration

The plugin owns the routed MoE/SwiGLU and shared gate-tail kernels. Current
hardware acceptance is pending after removal of eventfd completion and the
unified-dispatch refactor. The historical results and dependency boundary are
summarized in [the evidence record](../../../../../../tests/musa/acceptance_20260920.md).

## Baseline and ownership

The reference is the SGLang 0.5.11 MUSA vendor image, with core commit
`612785ffdcaf35552f1ed433a981d596ca9fe900` **plus its existing vendor changes**.
This is not an unmodified upstream SGLang checkout. The campaign added nine
core files on top; this candidate moves those additions into the plugin:

| Former core change | Plugin owner |
| --- | --- |
| `environ.py` capability flag | `impl/fused_moe.py`, uses guarded code defaults with environment overrides |
| MUSA FA3 capture metadata | `patches/fa3_graph_metadata.py` |
| Triton runner quant-info field | Existing MThreads fused-MoE dispatch, no dataclass ABI change |
| MoE sequence, M4 scheduling, M16K scratch | `moe/fused_moe.py` |
| Fused SwiGLU kernel and launcher | `moe/kernels.py` |
| Shared expert gate-tail kernel | `moe/shared_expert_gate_tail.py` |
| Unquantized post-load capability | Loaded-weight guards in `impl/fused_moe.py`, no weight mutation |
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

The following paths are enabled by default under their existing S5000,
model/shape/dtype/TP guards. They no longer need explicit `=1` exports; each
variable remains an override (`0`/`off` disables, `auto` uses the default):

- `SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE=1`: canonical-W13 fused prefill;
  the kernel sequence still disables fusion below 2048 tokens.
- `SGLANG_MUSA_M4_W13_BN64=1`: exact M4 up tile; original down/alignment config
  is not modified. The exact R2 M4 baseline resolver is also on by default;
  `SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE=0` restores the original resolver.
- `SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE=1`: reserve the 512 MiB
  fixed M16K routed-down scratch before KV-memory sizing, under the exact
  model/TP2/BF16 guard. The hook runs just before `init_memory_pool`; relative
  ordering against deterministic-mode setup is different from the former
  inline call and requires checking if that mode is enabled.
- `SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED=1`: narrow text-only M4 decode
  gate-tail path, with the original router, in-place addition and TP collective.

S5000 TP2 custom all-reduce uses 128 MiB by default. Other hardware/group
sizes retain 512 MiB. Payloads above the bound use the existing collective
fallback; `SGLANG_MUSA_CUSTOM_AR_MAX_SIZE_MB` overrides capacity explicitly.
The MUSA YAML includes native `index` and `mul` alongside copy/index-put, so
an explicit campaign blacklist is unnecessary.

Additional controls:

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

## Dispatch and validation

Routed MoE uses the existing `fused_moe` bridge and vendor implementation in
`impl/fused_moe.py`; there is no MoE-specific dispatcher. The model hooks only
supply execution context and preserve router/collective ownership.
`shared_expert_gate_tail`, `moe_sum_reduce` and `allreduce_rms_norm` are registered
with the common `OpManager`, including native reference implementations.
Common preference, per-op order and vendor allow/deny policies apply. These
bridges use `resolve_op` to avoid generic retry of a failed in-place kernel or
collective; guarded vendor fallbacks retain their existing semantics.
TopK and FLA continue through their existing registered operators; vendor
schedule/loader hooks remain implementation details of the pinned runtime.

Run [the validation handoff](../../../../../../tests/musa/README.md) on the
pinned image. Check actual dispatch hits, eager/refreshed graphs, shared BF16
materialization, two-stream join/reduction ownership and pre-KV M16K workspace
reuse. CPU tests do not establish device execution or throughput.
