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
construction; a full Docker image build was not performed.

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
and no task-owned inference processes remained after completion. Prefill graphs,
audio, a full Docker image build, and controlled performance benchmarking remain
outside this validation.
