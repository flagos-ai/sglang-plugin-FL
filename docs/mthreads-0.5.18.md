# MUSA empty runtime with SGLang 0.5.18

The current issue-resolution results are in
[the 2026-09-09 validation record](mthreads-0.5.18-validation.md).
The dated 2026-09-08 results below are the historical baseline before those fixes.

This setup upgrades the Moore Threads empty image used for Qwen3.6 while
retaining its MUSA 4.3.5, torch/torch_musa 2.9.0,
MATE/flash_attn_3 0.2.1 and FlagCX 0.13.0 stack. Upgrade the compiler to
FlagTree `0.6.2a3+mthreads3.6` (Triton 3.6) and FlagGems to master commit
`01433e8304d78ef6c16ce76fe68e51e16c0b4d66`, the latest snapshot selected
on 2026-09-08. The snapshot package version below is a local build label,
not an upstream release number. The optional
MUSA `sglang-kernel` 0.4.2 wheel is already present in this image and remains
necessary for the vendor MoE implementation. This is the document's hybrid
empty configuration, not a claim that every model runs without vendor kernels.

## Installation

Use `docker/mthreads/empty-0.5.18.containerfile`, or upgrade an isolated
container from its `BASE_IMAGE` manually:

```bash
export PATH=/root/.virtualenvs/sglang-0.5.6/bin:$PATH
export SGLANG_BUILD_RUST_EXTS=none
python -m pip uninstall -y triton flagtree
python -m pip install --no-deps 'flagtree==0.6.2a3+mthreads3.6' \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
mkdir -p /opt/FlagGems
curl -fL https://codeload.github.com/flagos-ai/FlagGems/tar.gz/01433e8304d78ef6c16ce76fe68e51e16c0b4d66 \
  -o /tmp/flaggems.tar.gz
tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.4.0.dev20260908+g01433e830 \
  python -m pip install --no-deps -e /opt/FlagGems
python -m pip install 'packaging>=26.0' PyYAML==6.0.1 sqlalchemy==2.0.48
export PYTHONPATH=/opt/FlagGems/src:${PYTHONPATH:-}
git clone --depth 1 --branch v0.5.18 https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_other.toml pyproject.toml
printf 'torch==2.9.0\ntorch_musa==2.9.0\n' > /tmp/musa-constraints.txt
python -m pip install -c /tmp/musa-constraints.txt '.[srt_empty]'
python -m pip install --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0
cd /path/to/sglang-plugin-FL
python -m pip install --no-deps .
```

SGLang 0.5.18 includes `srt_empty`; do not apply the 0.5.11 empty patches.
It imports xgrammar and compressed-tensors during server startup even for
unquantized models, so these are installed explicitly without their PyTorch
dependency chain. Rust extensions are disabled for this text/image setup.

The image also contains an older system-level `sglang_fl`. A regular install
into the inference virtual environment takes precedence. For editable
development, explicitly put the current plugin checkout on `PYTHONPATH` and
check `sglang_fl.__file__` before starting the server.

This FlagGems snapshot needs a source installation: its regular wheel omits
the `fused/DSA` namespace directory. Keep `/opt/FlagGems/src` first on
`PYTHONPATH`, because the base image also has an older system FlagGems that
can shadow an editable install. Check `flag_gems.__file__` and
`triton.__version__` in the actual inference Python before testing.

## MUSA CI image

The `ci` target in `docker/mthreads/empty-0.5.18.containerfile` adds the test
dependencies and the real ShareGPT cache used by the benchmark smoke tests.
It keeps the vendor Torch 2.9.0 stack, FlagTree 0.6.2a3+mthreads3.6 and the
FlagGems commit documented above. SGLang is installed from its v0.5.18 source
archive with explicit package-version metadata.

```bash
docker build --target ci \
  -f docker/mthreads/empty-0.5.18.containerfile \
  -t harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.5.18-musa-ci-20260909 .
```

The MUSA-only workflow `.github/workflows/musa-ci.yml` targets PRs into
`dev/0.5.18` and runs the existing unit, functional, E2E and benchmark entrypoints.
All scopes share one container: the first cold pull exceeded the old unit
job's 30-minute limit before any test began. The MUSA job allows 180 minutes
for initialization and the full suite, with separate limits and logs for each
test phase. A failing E2E group does not suppress the other two groups.
Configuration generation also uses the MUSA runner queue, with a temporary
Python environment; it does not depend on a separate GitHub-hosted runner.
Its setup script installs the current PR checkout with `--no-deps` and
checks the imported plugin path. The image supplies the dependencies, so
individual jobs do not upgrade SGLang, Torch, FlagTree or FlagGems.
At least four visible S5000 GPUs and the model mount `/data/models/Qwen`
are required for the configured TP4 cases. The cached benchmark dataset
works when GitHub Actions sets `HOME=/github/home`, without a network fetch.
The device check queries `mthreads-gmi` JSON and selects four idle devices
before importing Torch. It maps UUID allocations through the driver's visible
device report, respects existing numeric device masks, prefers a
contiguous group and exports `MUSA_VISIBLE_DEVICES` for all subsequent phases.
If fewer than four devices are idle, it waits up to 30 minutes instead of
running TP4 on occupied cards. The original model memory settings are retained.

Image preparation runs on the MUSA runner before starting the test container.
It uses Python's standard library to fetch the public Harbor image in 8 MiB
HTTP ranges, with at most four concurrent requests. Successful DNS resolutions
are cached for the duration of one pull without changing TLS hostname checks;
PAX archive headers support layers larger than 8 GiB. The manifest, configuration
and every compressed layer are checked against their pinned SHA256 digests.
The compressed archive is streamed into Docker, and the loaded image ID must
match the original configuration digest. The registry client and Docker import
command use direct connections. Existing verified images are reused; no registry login
or shared Docker daemon reconfiguration is required. All phases then execute
in one container, which is removed at job completion. Available transfer, device and
test logs are uploaded even when image preparation fails.

The shared graph fixture and graph arguments are compatible with 0.5.18:
`disable_cuda_graph` remains valid, while disabled prefill graphs use
`cuda_graph_backend_prefill: disabled`. Model cases and concurrency levels
are unchanged. See the current validation record for build and CI results.

## Serving

Select free physical GPU IDs with `MUSA_VISIBLE_DEVICES` before launch.

```bash
export FLAGCX_PATH=/sgl-workspace/FlagCX
export TORCH_COMPILE_DISABLE=1
export SGLANG_MUSA_FP32_TP_ALLREDUCE=1
export SGLANG_FL_PER_OP='silu_and_mul=flagos;mrotary_embedding=flagos;topk=reference;gemma_rms_norm=reference;fused_moe=vendor;chunk_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule_packed_decode=vendor'

python -m sglang.launch_server \
  --model-path /models/Qwen3.6-27B --tp-size 4 \
  --host 127.0.0.1 --port 30000 \
  --attention-backend fa3 --sampling-backend pytorch \
  --page-size 1 --disable-radix-cache --trust-remote-code \
  --context-length 4096 --chunked-prefill-size 2048 \
  --max-running-requests 8 --mem-fraction-static 0.65 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full --cuda-graph-max-bs 4
```

The first request may take several minutes while MATE compiles attention
kernels. Keep installations and server/test commands in named `tmux`
sessions on remote machines.

Use `/models/Qwen3.6-35B-A3B` to run the MoE model with the same settings.
For eager decode, replace the last line with
`--cuda-graph-backend-decode disabled`.

## Compatibility changes

- Unbridged fused operators fall back to `forward_musa`; FL registrations
  retain priority over the native fallback.
- Recurrent FLA preserves the old state-indexing arguments on kernels that
  accept them. New kernels omit unset arguments and reject unsupported
  nonempty indexing requests instead of silently changing their semantics.
- Triton 3.2 can resolve the two CUDA PDL names in disabled constexpr branches
  of upstream GDN normalization/convolution kernels. A live PDL call still
  fails compilation with an explicit MUSA error. Existing symbols are preserved.
- The vision FA3 entry point binds to the installed MUSA varlen implementation.
- PP ordering reads ranks from the new scheduler `ps` state, with the older
  direct rank field still supported. It also preserves upstream's skipped
  output communication for intermediate prefill chunks.
- MUSA's default FlagGems blacklist includes `broadcast_tensors`. The tested
  master snapshot mishandles zero-length dimensions, causing top-p sampling
  to crash when the top-k mask selects no tokens.
- Native `slice` handles the boolean buffers filled by SGLang 0.5.18's eager
  runner and preserves their view aliasing. The tested FlagGems snapshot
  rejects boolean slicing, which the original offline example exposed.
- The multimodal mask patch follows the module owning `get_embedding_and_mask`.
  In 0.5.18, `mm_utils` re-exports this function from `mm_schedule`; patching
  only `mm_utils._get_multimodal_mask` leaves the live implementation unchanged.
- Native `eq`, `eq_scalar`, and `equal` preserve integer placeholder IDs above
  `2**24`. FlagGems `01433e830` converts equality operands to FP32 and can
  incorrectly match adjacent IDs. Both contiguous and strided int32/int64
  inputs passed the live embedding-mask regression on MUSA.
  If overriding `SGLANG_FL_FLAGOS_BLACKLIST`, include `broadcast_tensors`,
  `slice`, `eq`, `eq_scalar`, and `equal` alongside your other required exclusions.
- `SGLANG_MUSA_FP32_TP_ALLREDUCE=1` also covers the active FlagCX communicator's
  three-dimensional vision tensors (`images, patches, channels`).
  Vision embedding and linear layers retain eagerly imported all-reduce
  references, bypassing the original source-module patch. The communicator
  fallback preserves in-place writes and the input dtype. This additional
  coverage leaves two-dimensional text tensors on their existing path;
  indiscriminate promotion regressed cross-node Dense TP4. The unsuccessful
  repeat/arange/copy blacklist experiments are not part of the default list.
- Qwen3.5/3.6 MTP uses synchronous scheduling on MUSA to avoid the overlap
  result-copy watchdog. Decode graphs remain enabled. The MTP example uses
  the effective scheduling mode and fresh prefixes in both comparison phases.
- A regular PP loop forwards an explicit `ShutdownReq` before exiting.
  Nonzero nodes skip HTTP initialization after their schedulers exit, and
  nonzero child exit codes still raise an error. Transport failures are not
  treated as successful shutdowns.

Run the targeted regression checks in the inference environment:

```bash
python -m pytest -q \
  tests/unit_tests/platform/test_musa_sglang_compat.py \
  tests/unit_tests/platform/test_musa_gated_layernorm.py \
  tests/unit_tests/platform/test_musa_triton_compat.py \
  tests/unit_tests/platform/test_musa_sampling_mask.py \
  tests/unit_tests/platform/test_musa_multimodal_mask.py \
  tests/unit_tests/platform/test_musa_runtime_config.py \
  tests/unit_tests/platform/test_musa_fp32_collective.py \
  tests/unit_tests/platform/test_musa_lifecycle.py \
  tests/unit_tests/platform/test_musa_pp_compat.py \
  tests/unit_tests/dispatch/test_base_fused_op_registration.py \
  tests/unit_tests/distributed/test_communicator.py \
  tests/unit_tests/distributed/test_communicator_hooks.py \
  tests/unit_tests/distributed/test_flagcx.py
```

## Additional model matrix

`tests/manual/musa_model_matrix.py` launches each model serially and checks
factual/arithmetic chat answers, sequential and four-concurrent 64-token
greedy decoding, four-concurrent top-p sampling, and decode graph capture
and replay. Qwen3.6 cases additionally check a red image. It records answers,
commands, versions, module locations and failures in JSON, with one server
log per model. These are serving smoke checks, not a model-quality benchmark.

With the serving environment above and a directory containing the model
subdirectories named in `CASES`, run inside `tmux`:

```bash
python tests/manual/musa_model_matrix.py \
  --model-root /models --output-dir /work/model-matrix \
  --large-model-tp 2 \
  --models phi4 gemma3 cohere qwen36_dense qwen36_moe
```

Phi-4-mini-instruct exercises `Phi3ForCausalLM`, rnj-1-instruct exercises
`Gemma3ForCausalLM`, and aya-23-8B exercises `CohereForCausalLM`. These are
architectures supported by upstream v0.5.18, beyond the Qwen examples;
this does not imply each architecture was first introduced in that release.

## Updated-stack validation (2026-09-08)

The matrix uses FlagTree `0.6.2a3+mthreads3.6` / Triton `3.6.0`, FlagGems
`01433e8304d78ef6c16ce76fe68e51e16c0b4d66` and unmodified SGLang `0.5.18`.
The actual imported FlagGems source location was checked. Runs used the
plugin's MUSA blacklist, including the new `broadcast_tensors` exclusion,
with FlagGems ATen replacement and FL fused-op dispatch enabled.

| Model | TP | Chat, 64-token decode, four-way greedy/sampling, graph replay |
| --- | --- | --- |
| Phi-4-mini-instruct | 1 | Passed; `Paris`, `221` |
| rnj-1-instruct (Gemma3) | 1 | Passed; `Paris`, `221` |
| aya-23-8B (Cohere) | 1 | Passed; `Paris`, `221` |
| Qwen3.6-27B | 2 | Passed; also identified the image as `red` |
| Qwen3.6-35B-A3B | 2 | Passed with vendor MoE; also identified the image as `red` |

The initial updated-stack regression run passed 40 tests. Later checks for
the example-driven fixes are reported below.
Two-rank FlagCX FP32/BF16 all-reduce also passed, with the communicator
explicitly checked to be active. The vendor MoE library reused its existing
tuning configuration because a Triton 3.6-specific configuration was absent;
no performance claim is made. The plugin wheel built successfully.

The additional matrix used two available S5000 GPUs on one host. The
separate cross-node TP/PP example runs are reported below.

## Original examples coverage (2026-09-08)

This section records the pre-fix results. References to unresolved issues
describe their state on that date; see the linked 2026-09-09 record for the
subsequent fixes and reruns.

All seven Python examples have been executed across the nine configurations
below, including the complete text, image and concurrent sections and the
MTP baseline. Execution coverage is complete; the MoE cross-node assertions,
MTP warning/default-overlap failure, and worker shutdown errors below prevent
an unconditional all-passed result. The independent model matrix above does
not substitute for these scripts.
All runs use the updated stack and TP2 unless specified below. Four images
were present: red square, cat, stop sign and digit seven.

| Original script | Required coverage | Observed result |
| --- | --- | --- |
| `qwen3_6_27b_offline_inference.py` | Two text prompts, four images | Passed, exit 0, after the boolean-slice fix |
| `qwen3_6_35b_a3b_offline_inference.py` | Two text prompts, four images | Passed, exit 0 |
| `qwen3_6_27b_concurrent.py` | `--mode all`, default 16 text requests, VL and mixed modes | Passed, exit 0, after rerunning a container-interrupted attempt |
| `qwen3_6_35b_a3b_concurrent.py` | `--mode all`, default 16 text requests, VL and mixed modes | Passed, exit 0 |
| `qwen3_6_27b_mtp_inference.py` | TP2, MTP plus baseline comparison, no skipped baseline | Eager and corrected graph run with `--disable-overlap-schedule`: each exit 0, 14 passed, 0 failed, 1 warning (9/12 exact baseline match). Default overlap+graph mode hit a 300-second watchdog timeout |
| `qwen3_6_27b_multinode.py` | TP2/PP2, text/VL and 32/8 concurrency | Passed after rank API fix: 10/10 checks, master exit 0; worker exit 247 after master shutdown |
| `qwen3_6_27b_multinode.py` | TP4/PP1, text/VL and 32/8 concurrency | 10/10 inference checks passed, master exit 0; worker exit 3 in post-scheduler HTTP startup (see below) |
| `qwen3_6_35b_a3b_multinode.py` | TP2/PP2, text/VL and 32/8 concurrency | First run 9/10 (`No Parking` for one stop sign), master 1 / worker 247. Full diagnostic rerun 10/10 plus 18/18 additional VL requests, master 0 / worker 247; the first failure remains unresolved |
| `qwen3_6_35b_a3b_multinode.py` | TP4/PP1, text/VL and 32/8 concurrency | First run and full diagnostic rerun each 9/10, text 32/32, VL 7/8 (`Red` for one stop sign); extra diagnostic VL 17/18; each master 1 / worker 3. Correctness failure reproduced |

The boolean-slice change passed a 42-test regression run and two-rank
FP32/BF16 FlagCX checks. Four additional PP compatibility tests passed,
covering old/new rank layouts, both rank parities, and skipped receives.
Several completed engine scripts printed multiprocessing resource-tracker
cleanup warnings after their assertions passed; these are retained in logs.
Container stops returned 137 with Docker `OOMKilled=false`; interrupted runs
have no successful script result and are not counted as passes.
An earlier worker retry on `moer_15` was stopped approximately three seconds
after starting, at 18:05 CST. Its waiting master was cleaned up by this task.
After the other workload released the worker GPUs, the task container
remained running on restart at 19:29 CST, allowing cross-node validation to
resume. Dense TP2/PP2 subsequently passed all ten checks, including 32/32
text requests and 8/8 VL requests across all four images. Its worker reported
a Gloo peer-disconnect error and exit 247 after the successful master shut
down; both nodes had no remaining test processes.

Dense TP4/PP1 also passed all ten inference checks, including 32/32 text and
8/8 VL requests. Its remote schedulers terminated with 0 after master
shutdown, but upstream's outer worker process then fell through to HTTP
startup with no tokenizer manager and returned 3 (`NoneType.server_args`).
This worker-lifecycle error and MCCL process-group cleanup warnings remain
visible in the logs; the master returned 0 and no test processes remained.

The Dense PP run logged 59 FlagGems autotuning candidate failures per node
when a temporary 24.68 GiB allocation exceeded the remaining memory at the
example's default static-memory fraction of 0.85. These were recovered
autotuning warnings; all requests and assertions completed. They are retained
in the logs and are distinct from a server crash or a failed assertion.

The first complete MoE TP2/PP2 run executed all six sections but failed the
concurrent VL aggregate: one of eight requests read `stop_sign.png` as
`No Parking`, although the image says `STOP`. The other seven VL requests
and all 32 concurrent text requests passed. This is a real assertion failure
at temperature 0, not a skipped image or a server-start failure. Recovered
autotuning OOM warnings numbered 47 on the master and 48 on the worker.
MoE TP4/PP1 also completed all six sections with 9/10 checks, text 32/32 and
VL 7/8: one stop-sign response was `Red`. It logged no autotuning OOM
warnings. Its worker showed the same post-scheduler HTTP startup error as
Dense TP4/PP1. Neither first MoE cross-node run is counted as a pass.

A diagnostic rerun imported the unchanged MoE example, logged every VL
answer, and retained all six original test sections, prompts, temperature 0,
concurrency and predicates. TP2/PP2 then passed 10/10, text 32/32 and VL 8/8.
Two additional sequential stop-sign requests and two more 8-way VL rounds
passed 18/18; all stop-sign replies were `Stop`. Master exit was 0 and worker
exit was 247. The rerun logged 18/17 recovered autotuning OOM warnings.
No model, dispatch or example fix intervened. The first failure did not
reproduce in this follow-up; its cause is not established and it is retained
as an intermittent correctness issue, rather than declared fixed.

The same diagnostic procedure on TP4/PP1 reproduced the original failure:
9/10 checks, text 32/32, VL 7/8, with a stop-sign reply of `Red` again. The
two extra sequential stop-sign requests passed; the two extra VL rounds
returned 7/8 and 8/8, respectively. The additional wrong answer was `.` for
a stop sign, giving 17/18 diagnostic requests correct overall. Master exit
was 1; worker exit was 3 with the same HTTP startup error. This rerun logged
11/12 recovered autotuning OOM warnings. The MoE TP4 correctness failure is
reproducible and unresolved; these results do not establish its root cause.

The MTP graph failure occurred after capture completed. Both scheduler ranks
remained in `process_batch_result_decode` at `result.copy_done.synchronize()`;
the scheduler logged `watchdog_timeout=300`. This is a failed graph test,
separate from the container-interrupted attempt. The example now exposes
`--disable-overlap-schedule` for applying the same scheduling mode to MTP and
baseline. With this option, all MTP prompts and the long generation completed;
the subsequent baseline exposed a cache assertion requiring `page_size=1`.
The MUSA baseline now uses that page size, consistent with the offline
examples. The full corrected run passed the script's checks with one
exact-match warning: 14 passed, 0 failed, 1 warning, exit 0. Both MTP and
baseline captured and replayed decode graphs, completed all 12 prompts and
generated 512 tokens in their throughput phases. Acceptance was `2.9792`;
exact outputs matched for 9/12 prompts (75%). Graphs remain enabled by
default, and the validation thresholds are unchanged. Use
`--disable-overlap-schedule` for the validated MUSA graph configuration;
overlap plus MTP graphs remains a known failure on this stack.

The full eager MTP run retained the default 12 prompts, 256-token prompt
limit, 512-token throughput generation and baseline phase. Acceptance was
`2.9346`, above the script's `2.0` threshold. Exact greedy outputs matched
baseline for 9/12 prompts (75%), so the script reported one warning while
its 14 other checks passed. The mismatches concerned gravity, states of
matter and the creative prompt; their cause has not been established.
This run preceded the baseline page-size correction above. Its measured
throughput is a diagnostic result, not a controlled performance benchmark.

Full logs and the per-attempt manifest are retained under
`/datapool/codex-musa-0518/examples-all/`; cross-node logs are under
`/datapool/codex-musa-0518/cross-node/`. Each retry uses a separate log or
result directory; an old exit-code file is not evidence for a later run.
The completed MTP logs are `dense-mtp-eager.log` and
`dense-mtp-graph-fixed.log`; both corresponding `.exit` files contain `0`.

The completed cross-node attempts below used plugin commit `d761f29`,
physical GPUs 0 and 1 on each of `moer_14` and `moer_15`, and the updated
stack above. Times are CST (UTC+8). Each directory contains `master.log`,
`worker.log`, per-role commands, start/finish times and Docker exit codes.

| Configuration | Result directory under `cross-node/` | Master start/end (CST) | Checks | Master/worker exits |
| --- | --- | --- | --- | --- |
| Dense TP2/PP2 | `dense-pp-r4-1930` | 19:31:39–19:37:42 | 10/10 | 0 / 247 |
| Dense TP4/PP1 | `dense-tp4-r1-1939` | 19:39:59–19:53:52 | 10/10 | 0 / 3 |
| MoE TP2/PP2 | `moe-pp-r1-1955` | 19:55:58–20:00:53 | 9/10 | 1 / 247 |
| MoE TP4/PP1 | `moe-tp4-r1-2005` | 20:05:00–20:13:54 | 9/10 | 1 / 3 |
| MoE TP2/PP2 diagnostic rerun | `moe-pp-diag-r2-2015` | 20:16:19–20:19:23 | Original 10/10; extra VL 18/18 | 0 / 247 |
| MoE TP4/PP1 diagnostic rerun | `moe-tp4-diag-r2-2020` | 20:21:42–20:31:54 | Original 9/10; extra VL 17/18 | 1 / 3 |

The diagnostic driver is retained at
`/datapool/codex-musa-0518/cross-vl-diagnostic.py`, alongside its launch
scripts. Its `VL_TRACE` and `VL_DIAGNOSTIC_RESULTS` records preserve the
individual answers. The imported MoE script's raw SHA256 was
`a8a7a81c24e9096918e2c0a366645825a2fc87c609bc4ac7aa7831c616b60b40`,
matching the local working copy byte-for-byte (CRLF line endings).
After the final run, both task containers contained only their idle
`sleep` process; no inference processes remained.

### Reproduce the original examples

Use the inference environment and dispatch configuration above. From the
plugin checkout, select two free GPUs, set `TP_SIZE=2`, and require all images
before running either model's offline and concurrent scripts:

```bash
export MUSA_VISIBLE_DEVICES=0,1
export MTHREADS_VISIBLE_DEVICES=0,1
export TP_SIZE=2
export IMAGE_DIR="$PWD/examples/test_images"
for image in red_square.jpg cat.jpg digit_seven.png stop_sign.png; do
  test -f "$IMAGE_DIR/$image" || exit 1
done

MODEL_PATH=/models/Qwen3.6-27B python examples/qwen3_6_27b_offline_inference.py
MODEL_PATH=/models/Qwen3.6-27B python examples/qwen3_6_27b_concurrent.py --mode all
MODEL_PATH=/models/Qwen3.6-35B-A3B python examples/qwen3_6_35b_a3b_offline_inference.py
MODEL_PATH=/models/Qwen3.6-35B-A3B python examples/qwen3_6_35b_a3b_concurrent.py --mode all

# Default graph configuration: the MUSA plugin selects synchronous MTP scheduling.
MODEL_PATH=/models/Qwen3.6-27B python examples/qwen3_6_27b_mtp_inference.py
# Eager comparison: retains all prompts and baseline.
MODEL_PATH=/models/Qwen3.6-27B python examples/qwen3_6_27b_mtp_inference.py \
  --disable-cuda-graph --disable-piecewise-cuda-graph
# Graph comparison without overlap, including the MUSA baseline page-size fix.
MODEL_PATH=/models/Qwen3.6-27B python examples/qwen3_6_27b_mtp_inference.py \
  --disable-overlap-schedule
```

For each multinode script, run matching commands in `tmux` on both hosts,
using the same model path, ports and two visible GPUs per host. Set `ROLE`
to `master` on the first host and `worker` on the second. Set `MASTER_ADDR`
to the first host's reachable address. Set the Gloo, MCCL and FlagCX socket
interfaces to the interface connecting the hosts (`bond0` in this setup).
Run each configuration separately after the previous processes exit:

```bash
# Repeat with qwen3_6_35b_a3b_multinode.py and its MODEL_PATH.
export MODEL_PATH=/models/Qwen3.6-27B
python examples/qwen3_6_27b_multinode.py \
  --role "$ROLE" --master-addr "$MASTER_ADDR" \
  --tp 2 --pp 2 --port 31828 --dist-port 32828 --nccl-port 33828 \
  --max-wait 1200 --request-timeout 600

python examples/qwen3_6_27b_multinode.py \
  --role "$ROLE" --master-addr "$MASTER_ADDR" \
  --tp 4 --pp 1 --port 31828 --dist-port 32828 --nccl-port 33828 \
  --max-wait 1200 --request-timeout 600
```

Keep the default 16-way single-node concurrency, 32-way cross-node text
concurrency, 8-way cross-node VL concurrency and MTP baseline comparison.
Missing images, skipped cases or server-start failures do not establish
complete example coverage.

## Initial baseline validation (2026-09-08)

Before the FlagTree/FlagGems upgrade, validated on four MTT S5000 80 GB GPUs
with Triton 3.2.0 / FlagGems 5.3.0rc2 from the base image and the
unmodified upstream SGLang v0.5.18 source. FlagGems ATen replacement and FL
fused-op dispatch were enabled throughout model validation.

| Case | Result |
| --- | --- |
| Qwen3-0.6B, TP1, eager | Sequential + four concurrent requests; 16 output tokens each |
| Qwen3.6-27B, TP4, eager | Text, image, four concurrent chat requests correct |
| Qwen3.6-27B, TP4, decode graph | Captured batch sizes 1/2/4 on all ranks; text/image/concurrent chat correct; sequential + four concurrent 64-token decode requests passed with graph replay |
| Qwen3.6-35B-A3B, TP4, decode graph | Same graph, chat and 64-token decode checks passed using vendor MoE |
| FlagCX, four ranks | Active communicator asserted; FP32/BF16 all-reduce matched expected sums on every rank |
| Regression suite before adding the sampling-mask cases | 37 passed, including 16 GPU normalization cases and rejection of live PDL |
| Packaging | Plugin wheel built successfully |

Chat checks asked for France's capital and the color of a red image, and
verified `Paris` / `red` in the actual responses, with HTTP 200 and positive
completion-token counts. The 64-token checks forced continued decoding to
exercise graph replay beyond the short factual answers.

This initial older-stack baseline did not cover cross-node TP/PP or
speculative decoding; the updated-stack example results above cover those
paths. Prefill graphs, audio and controlled performance benchmarking remain
outside the completed validation. This initial baseline used a manually
prepared container; the later full CI image build is recorded in the current
validation record.
