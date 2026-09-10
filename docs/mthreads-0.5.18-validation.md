# Moore Threads validation follow-up (2026-09-09)

This record follows the failures found while executing all seven original
examples against unmodified SGLang v0.5.18 on `moer_14` and `moer_15`.
The [setup guide](mthreads-0.5.18.md) contains installation and reproduction
commands, additional model coverage, and the historical 2026-09-08 results.

All seven scripts, covering nine configurations, completed their full test
bodies. The four cross-node configurations passed with both roles exiting 0.
MTP completed with **14 passed, 0 failed, 1 warning**: strict text equality
against its baseline remains **9/12 (75%)**, below the unchanged 90% threshold.
This record does not classify that warning as resolved.

The compiler and operator sources remain FlagTree `0.6.2a3+mthreads3.6`
(Triton 3.6.0) and FlagGems
`01433e8304d78ef6c16ce76fe68e51e16c0b4d66`, selected on 2026-09-08.
PyTorch/torch_musa are 2.9.0, MUSA is 4.3.5, and FlagCX is 0.13.0.
Tests use S5000 GPUs, two GPUs per host for cross-node cases, and the
documented vendor MoE/FLA paths. Diagnostic runs that disable FlagGems are
explicitly identified below and are not substitutes for enabled-FlagGems tests.

## Corrections and evidence

### Multimodal placeholders

SGLang moved the live mask helper to `mm_schedule`; patching its old
`mm_utils` re-export did not change the function used for embedding assembly.
The plugin now patches the module owning `get_embedding_and_mask`.
FlagGems equality also casts integer operands to FP32, so adjacent IDs above
`2**24` can compare equal. Native `eq`, `eq_scalar`, and `equal` preserve these
IDs. GPU tests cover int32/int64, contiguous/strided inputs, and the actual
embedding entry point. These independent fixes alone did not eliminate the
intermittent cross-node MoE VL failure.

### MoE cross-node VL correctness

The FP32 TP reduction option did not cover eagerly imported functions in
vision `VocabParallelEmbedding` and `RowParallelLinear`. Instrumentation
recorded BF16 reductions with shapes `(4, 256, 1152)` and `(1, 256, 1152)`
reaching the active FlagCX hook. A patch on `GroupCoordinator` alone was also
bypassed by that hook. The fix covers `CommunicatorFL.all_reduce` for the
three-dimensional vision layout (`images, patches, channels`), preserving its
in-place return contract and converting the result back to the input dtype.
Existing source-level FP32 reductions are unchanged; the extra hook does not
promote one- or two-dimensional text tensors.

With FlagGems enabled and the original overlap scheduler retained, the
vision-only candidate passed the full TP4 example, 10/10 checks plus 18/18
extra VL requests, with both roles exiting 0. A separately started final
run then passed 10/10 plus **82/82** extra VL requests (two sequential stop
signs and ten rounds of eight concurrent requests), again with exits 0/0.
Prompts, temperature 0, images, predicates, and the original 32-text/8-VL
concurrency were unchanged.

The option remains `SGLANG_MUSA_FP32_TP_ALLREDUCE=1`, as in the setup guide.
It trades additional conversion and communication volume for accuracy.

### MTP overlap watchdog

Default overlap plus decode graphs stalled at `result.copy_done.synchronize()`.
The failure reproduced after changing RMSNorm and after placing copies on the
forward stream. The plugin now selects synchronous scheduling for Qwen3.5/3.6
MTP on MUSA. Graph capture/replay and all original validation prompts remain
enabled. The example reads the effective scheduling mode from the MTP engine
and uses it in the baseline, with radix caching disabled in both phases.

The candidate completed all 12 MTP prompts, all 12 baseline prompts, and both
512-token throughput phases: 14 checks passed, zero failed, one strict-output
warning (9/12 exact matches). Native ATen plus reference SiLU/vendor RMSNorm
also produced 9/12; explicitly computing the output layer in FP32 changed the
divergent prompts but also produced 9/12. Neither became a default workaround.
The original 90% comparison threshold was retained.

SGLang documents that varying batch shapes can change floating-point reduction
orders even at temperature 0; see its
[deterministic inference documentation](https://docs.sglang.io/docs/advanced_features/deterministic_inference).
This explains a possible mechanism, rather than establishing every divergent
token's cause. An isolated deterministic-mode prototype added the missing
MUSA `PrivateUse1` dispatch selection and verified batch-invariant matrix
multiplication independently. Its full MTP example still reported 14/0/1,
with 8/12 exact matches (67%). It did not improve this warning and is not
included in the plugin. The final normal graph run again reported 9/12.
The three differences concern the gravity explanation, states of matter,
and robot story; the other nine prompts match exactly. The original content,
acceptance-length and throughput checks passed. Strict output equality
remains unresolved; the differing answers are retained in `MTP_RESULT_JSON`.

### Worker shutdown

The regular PP loop never checked `gracefully_exit`. It now forwards an
explicit `ShutdownReq`, waits for that request's send, and unwinds the regular
PP loop. Other scheduler loops retain their existing handlers. Communication
exceptions continue through SGLang's error path.

After nonzero-node schedulers exited, the outer HTTP launcher attempted to
initialize serving with a missing tokenizer. It now skips that HTTP step.
The scheduler wait checks child exit codes and raises for nonzero exits;
failed workers are not converted into successful runs.

The final Dense and MoE TP2/PP2 runs each completed 10/10 original checks
plus 18/18 extra VL requests, with master **0** and worker **0**. Dense and
MoE TP4/PP1 also exited 0/0 after their full original and extra VL checks.

### Scope of additional FP32 communication

Promoting every BF16 tensor reaching the active communicator introduced a
Dense TP4 regression. Errors appeared at `repeat`/`arange` autotuning,
scalar synchronization, dtype conversion, and collective launch. The driver
also recorded firmware resets. These asynchronous reporting locations did
not establish individual operator defects.

The `repeat`/`arange` blacklist passed once on GPUs 6/7 but failed on GPUs 2/3.
A `to_copy`/`copy_` blacklist also failed on GPUs 2/3. Native ATen passed the
32-text section but failed during eight concurrent VL requests. Disabling
overlap did not resolve the failure. All these exclusions were withdrawn.

Independent four-rank collectives passed FP32/BF16 reductions from 16 to
41,943,040 elements and 20 irregular batch sizes. The first 5,230 comparable
model collectives also had matching shape, dtype and layer ordering on every
rank. These checks narrow the investigation but do not establish the
underlying MUSA/FlagCX runtime defect.

The additional FP32 hook now covers only the observed missing vision layout.
Omitting the overbroad hook passed Dense TP4 10/10 plus 18/18, exits 0/0.
The vision-only candidate preserved the MoE VL fix. Two independently
started Dense TP4 runs, on GPUs 0/1 and on the previously failing GPUs 2/3,
each passed 10/10 plus 18/18, with both roles exiting 0. Original FlagGems
`repeat`, `arange`, dtype conversion, and copy implementations remain enabled.

The final runtime blacklist is `broadcast_tensors`, `slice`, `eq`,
`eq_scalar`, `equal`, `count_nonzero`, `cumsum`, `mm`, `unique`, `_unique2`,
`unique_dim`, and `unique_consecutive`. The CI override retains all twelve
runtime exclusions and its pre-existing `mul`/`sort` exclusions; the override
replaces the runtime list rather than extending it.

## Original-example follow-up coverage

| Example/configuration | Coverage | Follow-up result |
| --- | --- | --- |
| `qwen3_6_27b_offline_inference.py`, TP2 | Two text prompts and four images | All validations passed, exit 0 |
| `qwen3_6_35b_a3b_offline_inference.py`, TP2 | Two text prompts and four images | All validations passed, exit 0 |
| `qwen3_6_27b_concurrent.py`, TP2 | `--mode all`: 16 text, 16 VL, 16 mixed requests | All validations passed, exit 0 |
| `qwen3_6_35b_a3b_concurrent.py`, TP2 | `--mode all`: 16 text, 16 VL, 16 mixed requests | All validations passed, exit 0 |
| `qwen3_6_27b_mtp_inference.py`, TP2 | All 12+12 prompts, baseline, graphs, both 512-token phases | 14 passed / 0 failed / 1 warning; exit 0 |
| `qwen3_6_27b_multinode.py`, TP2/PP2 | All six sections, 32 text / 8 VL concurrency, extra VL | 10/10 plus 18/18; master/worker 0/0 |
| `qwen3_6_27b_multinode.py`, TP4/PP1 | Same complete coverage; repeated on two GPU pairs | Each 10/10 plus 18/18; master/worker 0/0 |
| `qwen3_6_35b_a3b_multinode.py`, TP2/PP2 | All six sections, 32 text / 8 VL concurrency, extra VL | 10/10 plus 18/18; master/worker 0/0 |
| `qwen3_6_35b_a3b_multinode.py`, TP4/PP1 | Same complete coverage, independently repeated with ten extra VL rounds | 10/10 plus 18/18; repeated 10/10 plus 82/82; both runs 0/0 |

All paths are under `examples/`. The cross-node scripts were executed through
a diagnostic import that retains their full test body. Extra VL requests run
only after that body finishes. No baseline, image, or concurrency case was
removed to obtain a pass. Cross-node examples retain overlap scheduling,
radix caching, their default 8192 chunked prefill size, and eager execution.
The MTP example retains decode graph capture/replay, with the documented
synchronous scheduling fallback.

The final targeted compatibility/distributed regression run passed **87 tests
in 20.44 seconds**. Three unchanged fused-op registration tests also passed
in a separate run. Ruff and `git diff --check` passed. A wheel built from the
final package source passed content checks for the runtime/lifecycle modules,
vision-only FP32 scope, and the retained blacklist. This verifies package
construction. The subsequent full CI image build is recorded below.

## MUSA CI environment follow-up (2026-09-09)

The `ci` target of `docker/mthreads/empty-0.5.18.containerfile` built
successfully from a clean source context on `moer_14`. Its runtime is SGLang
0.5.18, Torch/torch_musa 2.9.0, FlagTree 0.6.2a3+mthreads3.6 / Triton 3.6.0,
and FlagGems 5.4.0.dev20260908+g01433e830. The CI stage includes pytest and
the real ShareGPT dataset cache. Dependency installation preserves the
vendor Torch version, and the SGLang source archive receives explicit 0.5.18
package-version metadata.

The public image is
`harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.5.18-musa-ci-20260909`,
published with digest
`sha256:8c31e56f542ed1e7d2a07bf95c241b7700eb4945014cdc48da73dd77c261207b`.
An empty Docker credential configuration successfully reads its manifest.
The MUSA CI configuration pins this digest.

The existing `ci.yml` / `_platform_test.yml` pipeline enables MUSA alongside
NVIDIA on `dev/0.5.18` and runs the complete MUSA test matrix. Its setup
installs the current PR checkout without resolving
dependencies and checks that `sglang_fl` imports from that checkout.
The approved shared test corrections initialize the graph fixture's vendor
name and replace obsolete `disable_piecewise_cuda_graph` arguments with
`cuda_graph_backend_prefill: disabled`. The existing `disable_cuda_graph`
argument remains valid and is retained.

Fresh-container preflight uses four S5000 GPUs on `moer_14`, the repository
mounted at `/workspace`, models under `/data/models/Qwen`, and the Actions
home directory `/github/home`. It imports FlagGems from `/opt/FlagGems/src`.

| Preflight scope | Result |
| --- | --- |
| Unit | 282 passed, exit 0 |
| Functional | 36 passed / 3 existing skips, exit 0 |
| Inference | Qwen3-4B TP2, Qwen3-0.6B TP1, Qwen3.6-35B-A3B TP4 and Qwen3.6-27B TP4: all four passed, exit 0 |
| Concurrent | Both Qwen3.6 TP4 cases passed all text, VL and mixed requests, exit 0 |
| Serving | Qwen3-4B TP2 and both Qwen3.6 TP4 cases: 5 checks each, all 15 passed, exit 0 |
| Benchmark | Original throughput, latency and serve smoke tests: all three passed, exit 0 |

These preflight results are separate from the GitHub Actions result.
The authoritative CI outcome is published on the
[PR #95 checks page](https://github.com/flagos-ai/sglang-plugin-FL/pull/95/checks)
under `CI` / `test-musa` / `MUSA full test suite`.
Before integration, [standalone run 34379124461](https://github.com/flagos-ai/sglang-plugin-FL/actions/runs/34379124461) at `9876d93` passed 22
helpers, 282 unit tests, 36 functional tests (three existing skips), all four
inference and both concurrent configurations, all 15 serving checks, and all
three benchmark smoke cases. Its full job took 51m30s, including successful
container cleanup and artifact upload. It reused the verified image cache.
### Integration with the current upstream CI (2026-09-10)

The PR is being rebased onto `upstream/dev/0.5.18` at `0291457` (NVIDIA CI
#99). MUSA is enabled in the existing platform registry and `_platform_test.yml`;
there is no separate MUSA event workflow. Other platforms retain their original
job bodies. MUSA runs all six test scopes in one native Actions job container
with a 180-minute limit, avoiding repeated initialization of the 30.05 GiB
image. The shared notification uses that job's result. An E2E failure does not
suppress its peer groups, and each group retains its own log and timeout.

The custom registry downloader, Docker lifecycle wrapper and transport probes
have been removed. Docker now pulls the pinned image, and Actions manages the
container lifecycle. The retained MUSA scripts install the checkout without
changing the image's dependencies and select four idle GPUs within the runner's
existing allocation. Nine selector regression checks cover occupancy, numeric
and UUID allocations, allocation intersection and insufficient capacity.
The existing model sizes, prompts, concurrency and assertions are unchanged.
`examples/README.md` matches upstream; backend-specific results belong here.

The completed historical run linked above predates this integration and rebase.
The current head's outcome must be checked separately on the PR checks page.

### CI failures addressed and remaining infrastructure limit

- The original 30-minute unit job timed out while initializing the image, before
  tests began. MUSA now initializes once for the complete suite.
- Run `34331550776` passed unit and functional tests but encountered an occupied
  GPU (66,325 MiB used). Selection now waits for four idle allocated devices.
  Live probes preserved numeric and noncontiguous allocations and exercised a
  tensor operation on each selected logical device. UUID allocations observed
  in run `34360228140` are mapped through the driver's visible device report.
- Run `34369228544` passed three of four inference cases, both concurrent cases
  and all 15 serving checks. Qwen3-4B at temperature 0.7 returned a Chinese name
  for Paris and failed the existing literal `Paris` assertion. The MUSA-only
  engine override fixes `random_seed=42`; three independent preflight engines
  and run `34379124461` then passed with the original prompts and assertions.
- Cold image transfers remain sensitive to runner bandwidth. Both native Docker
  initialization and experimental transfer clients timed out on cold runners.
  The range client still transferred only 5.87 GiB in 60 minutes in run
  `34360832372`, despite a successful 30.05 GiB local preflight. Those experiments
  did not fix sustained CI transfer throughput and are not production CI code.
  The successful 51m30s run reused a verified cache and does not validate a cold
  pull. Earlier failed runs remain failures, with no GPU-test pass inferred.

Build and phase logs, timestamps and exit codes remain under
`/datapool/codex-musa-0518/ci-0909/`. The historical successful-run evidence is
also archived as `ci-0909/pr95-musa-ci-9876d93-evidence.tar.gz`, SHA256
`116ca72f0d2fea1873825e84fa5840ceecf11d383b245ffe7c99022cc74ba9f5`.
No test case was removed or newly skipped to obtain these results.

## Selected evidence locations

Paths are relative to `/datapool/codex-musa-0518/`. Cross-node directories
contain each role's log, command, start/finish timestamps and Docker exit code.
The final manifest, `issues-0909/release-validation-manifest.json`, records
results and log SHA256 values. Times below are CST (UTC+8), 2026-09-09.

| Final run under `cross-node/` | Master start/end | Original + extra VL | Master/worker |
| --- | --- | --- | --- |
| `dense-tp4-vision-fp32-0909-1401/` | 14:01:00–14:19:15 | 10/10 + 18/18, GPUs 0/1 | 0/0 |
| `dense-tp4-vision-retry-0909-1404/` | 14:03:03–14:21:08 | 10/10 + 18/18, GPUs 2/3 | 0/0 |
| `moe-tp4-vision-fp32-0909-1346/` | 13:46:39–13:52:50 | 10/10 + 18/18 | 0/0 |
| `release-dense-pp-0909-1417/` | 14:19:20–14:21:40 | 10/10 + 18/18 | 0/0 |
| `release-moe-pp-0909-1417/` | 14:21:40–14:24:08 | 10/10 + 18/18 | 0/0 |
| `release-moe-tp4-endurance-0909-1417/` | 14:21:10–14:34:48 | 10/10 + 82/82 | 0/0 |

| Other final evidence | Result |
| --- | --- |
| `issues-0909/release-examples/` | All four original single-node scripts, exit 0 each, 14:12–14:24 |
| `issues-0909/mtp-release-0909-1413.log` | Original graph MTP plus baseline, 14/0/1, 9/12 exact, exit 0, 14:13–14:23 |
| `issues-0909/release-regression.log` | 87 passed in 20.44 seconds |
| `issues-0909/final-packaging.log` | Three unchanged fused-op registration tests passed separately |
| `issues-0909/release-packaging.log` | Final-source wheel built and its contents verified |
| `issues-0909/release-wheel/sglang_fl-0.1.0-py3-none-any.whl` | SHA256 `65816b2aa7cb82f80cd773b7de2b5cf701e9e7f2e75efb0ead5ad2e7bf815745` |

The final source is retained in `issues-0909/package-source.tar`; its Python
syntax trees match the tested `issues-0909/plugin-release` candidate after
excluding comments/docstrings, and its runtime YAML is identical. The earlier
vision-only candidate differs only in comments, logging text, and formatting
of the same condition. Final evidence, scripts and the wheel are also backed
up locally in `musa-validation/issues-0909/release-evidence/`.

Selected negative controls remain separate from successful runs:

| Diagnostic evidence | Finding |
| --- | --- |
| `cross-node/dense-tp4-meta-retry-0909-1311/` | Repeat/arange exclusions did not fix GPUs 2/3; 5/10, exits 1/247 |
| `cross-node/dense-tp4-copies-0909-1353/` | Cast/copy exclusions also failed on GPUs 2/3; exits 1/247 |
| `cross-node/dense-tp4-native-0909-1341/` | Native ATen passed text 32/32 but failed concurrent VL; exits 1/247 |
| `cross-node/dense-tp4-comm-trace-0909-1341/` | Collective JSONL and scheduler stacks from the failed broad-FP32 attempt |
| `cross-node/dense-tp4-no-comm-fp32-0909-1333/` | Source-only FP32 control passed 10/10 + 18/18, exits 0/0 |
| `issues-0909/collective-sizes-master.log` | Four-rank FP32/BF16 regular-size probe passed |
| `issues-0909/collective-irregular-master.log` | Four-rank irregular-size probe passed |
| `issues-0909/mtp-native-0909-1216.log` | Native-operator control, 14/0/1, 9/12 exact |
| `issues-0909/mtp-lm-fp32-cast-0909-1159.log` | FP32 output-layer control, 14/0/1, 9/12 exact |
| `issues-0909/mtp-deterministic-0909-1242.log` | Deterministic-mode prototype, 14/0/1, 8/12 exact; not shipped |

Unsuccessful overlap, cache, WAR-barrier, full-device synchronization and
RMSNorm controls are also retained in separate directories. They were not
shipped as MoE mitigations. Older prototype regression counts do not describe
the final source; the final scoped suite has 87 passing tests.

## Limits

The MTP overlap path is avoided, not repaired inside the MUSA runtime.
Strict MTP baseline equality is still reported independently of content
validation. The model checks are serving smoke tests, not quality benchmarks.
Resource-tracker cleanup `KeyError` warnings and MCCL process-group teardown
warnings remain in some otherwise successful logs; final exit codes were 0
and no task-owned inference processes remained after those completed runs.
Prefill graphs, audio and controlled performance benchmarking remain outside
this validation. Benchmark smoke tests check entrypoint execution, not
performance regressions.
