# Qwen3.6 examples and Ascend acceptance

These examples exercise `sglang-plugin-FL` with Qwen3.6-27B and
Qwen3.6-35B-A3B. They are executable correctness checks: a missing model,
missing/empty test image, invalid response, or child-process failure returns a
non-zero exit code.

The Ascend commands below target SGLang 0.5.18. They describe the acceptance
matrix and artifact format; they do not claim measured performance results.

## Examples

| Script | Coverage | Ascend default |
| --- | --- | --- |
| `qwen3_6_27b_offline_inference.py` | Sequential text and vision correctness | TP=4 |
| `qwen3_6_35b_a3b_offline_inference.py` | Sequential text and vision correctness | TP=4 |
| `qwen3_6_27b_concurrent.py` | Concurrent text, vision, and mixed requests | TP=4 |
| `qwen3_6_35b_a3b_concurrent.py` | Concurrent text, vision, and mixed requests | TP=4 |
| `qwen3_6_27b_mtp_inference.py` | MTP generation and optional baseline comparison | TP=4 |
| `qwen3_6_27b_multinode.py` | Two-node server plus text/vision validation | CLI-controlled |
| `qwen3_6_35b_a3b_multinode.py` | Two-node server plus text/vision validation | CLI-controlled |

The Hygon MTP profile uses TP=2; see the Hygon section below.

Ascend is detected through `torch.npu`. The examples select `device=npu`,
`attention_backend=ascend`, and BF16 without assuming CUDA. MTP cache cleanup
also uses the active accelerator rather than calling CUDA unconditionally.

All vision-capable examples require these non-empty files under
`IMAGE_DIR` (default `examples/test_images`):

- `red_square.jpg`
- `cat.jpg`
- `stop_sign.png`
- `digit_seven.png`

## Full single-node acceptance

The entrypoint below verifies both models. For each model it runs offline and
all concurrent modes at TP=4, then recreates a TP=2 engine for each of three
text-concurrency canary rounds. It also compares Qwen3.6-27B MTP with the
baseline at TP=4. Correctness mode disables the Ascend overlap plan stream;
the MTP comparison also explicitly disables overlap scheduling.
The 27B MTP example enforces this validated eager/synchronous mode whenever it
detects Ascend, so a direct invocation cannot silently enter graph replay.

Because this is a long-running GPU job, run it in `tmux`:

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-single \
  'bash scripts/ascend/run_single_node_acceptance.sh \
   --result-dir /tmp/sglang-fl-acceptance/single'
tmux capture-pane -t ascend-single -p
```

Default model paths can be overridden without editing the scripts:

```bash
MODEL_27B_PATH=/models/Qwen3.6-27B \
MODEL_35B_PATH=/models/Qwen3.6-35B-A3B \
bash scripts/ascend/run_single_node_acceptance.sh
```

`--model 27b` or `--model 35b` selects a partial run. Omitting `--model` is the
full acceptance run.

## Two-node TP=4, PP=2 examples

Both hosts need the same checkout, model paths, images, and four visible NPUs.
Replace the address and interface below with the routable HCCL network values.
Start both sessions close together; the master is rank 0 and the worker is rank
1. The default `--model all` runs 27B and then 35B-A3B with distinct ports.

Master node:

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-multinode-examples \
  'bash scripts/ascend/run_multinode_examples.sh \
   --role master --master-addr 192.168.1.10 --interface eth0 \
   --result-dir /tmp/sglang-fl-acceptance/examples-master'
```

Worker node:

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-multinode-examples \
  'bash scripts/ascend/run_multinode_examples.sh \
   --role worker --master-addr 192.168.1.10 --interface eth0 \
   --result-dir /tmp/sglang-fl-acceptance/examples-worker'
```

Inspect either session with:

```bash
tmux capture-pane -t ascend-multinode-examples -p
```

The wrapper fixes the topology at `tp_size=4`, `pp_size=2`, `nnodes=2` and
fails on an unsupported exit. The worker only normalizes the narrowly
identified SGLang 0.5.18 clean-shutdown path; arbitrary exit code 3 or an
incomplete scheduler shutdown still fails.

## Two-node serving benchmark

Run the benchmark wrapper in separate `tmux` sessions on the same two hosts.
It starts an SGLang 0.5.18 server with TP=4 and PP=2, then the master executes
the fixed matrix for both 27B and 35B-A3B:

| Input tokens | Output tokens | Requests | Max concurrency | Runs used |
| ---: | ---: | ---: | ---: | --- |
| 1,024 | 1,024 | 64 | 64 | 2-4 of 4 |
| 4,096 | 1,024 | 64 | 64 | 2-4 of 4 |
| 16,384 | 1,024 | 64 | 64 | 2-4 of 4 |

Master node:

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-benchmark \
  'bash scripts/ascend/run_multinode_benchmark.sh \
   --role master --master-addr 192.168.1.10 --interface eth0 \
   --result-dir /tmp/sglang-fl-acceptance/benchmark-master'
```

Worker node:

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-benchmark \
  'bash scripts/ascend/run_multinode_benchmark.sh \
   --role worker --master-addr 192.168.1.10 --interface eth0 \
   --result-dir /tmp/sglang-fl-acceptance/benchmark-worker'
```

The driver uses the official 0.5.18 entrypoint
`python -m sglang.benchmark.serving` and one unique `--output-file` per run.
`random-ids`, `--random-range-ratio 1.0`, and `--tokenize-prompt` make the
requested lengths exact. Every JSONL record must report all 64 requests, exact
input/output totals, 64 exact per-request lengths, no request errors, and all
required finite latency/throughput metrics. A CSV row is never presented as a
passing result unless those checks succeed.

Artifacts for each model contain:

- `configuration.json`, including the exact client arguments;
- twelve official JSONL records and twelve captured client logs;
- `raw_runs.csv`, retaining all four runs per shape;
- `summary.csv`, averaging only runs 2-4;
- `failures.txt` when any shape fails.

You can point the driver at an already-running server directly:

```bash
python3 benchmarks/benchmark_throughput_serve.py \
  --model /models/Qwen3.6-35B-A3B \
  --model-name qwen3_6_35b_a3b \
  --host 127.0.0.1 --port 30000 \
  --output-dir /tmp/sglang-fl-acceptance/manual-benchmark
```

## Runtime settings

The wrappers preserve explicit caller overrides and otherwise set the plugin,
FlagCX, per-op dispatch, HCCL buffer, and visible-device defaults used for the
Ascend acceptance environment. Useful overrides are:

| Variable | Default |
| --- | --- |
| `MODEL_27B_PATH` | `/models/Qwen3.6-27B` |
| `MODEL_35B_PATH` | `/models/Qwen3.6-35B-A3B` |
| `IMAGE_DIR` | `examples/test_images` |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` |
| `SGLANG_FL_DIST_BACKEND` | `flagcx` |
| `FLAGCX_PATH` | `/opt/FlagCX` |
| `PYTHON_BIN` | `python3` |

Each wrapper writes its command lines, output logs, and environment/package
manifest to its result directory. A final `PASS` line means every requested
stage returned zero; absence of that line is not a pass.

## Manual example use

Individual examples remain usable outside the full matrix:

```bash
MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=4 \
python3 examples/qwen3_6_27b_concurrent.py --mode all

MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=4 \
python3 examples/qwen3_6_27b_mtp_inference.py \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --disable-overlap-schedule
```

For precision comparison across plugin modes, use
`tests/test_precision_align.py` or `tests/validate.sh`.

## Hygon MTP (single-node eager validated)

[PR #149](https://github.com/flagos-ai/sglang-plugin-FL/pull/149) adds a Hygon path
for original BF16 Qwen3.6-27B and Qwen3.6-35B-A3B checkpoints with complete MTP
heads. Runtime commit `c46d2258b2362f0ea171b30781683138f8c8de1b` passed the manual
checks below; the PR is not merged. Rebuild from that checkout, or mount and
install it into the prepared 0.5.18 Hygon image with
`python -m pip install --no-deps --no-build-isolation -e .`.
The published CI image alone does not contain these new plugin changes. See the
[Hygon setup and MTP guide](../docker/hygon/README.md) for vendor-stack checks and
read-only model mounts, including the separate 35B host directory.

Inside the prepared container, run sequentially from a host `tmux` session with
the two assigned, idle devices in `HIP_VISIBLE_DEVICES`:

```bash
set -euo pipefail
cd /opt/sglang-plugin-fl
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
: "${HIP_VISIBLE_DEVICES:?Set the two assigned Hygon devices first}"
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DIST_BACKEND=flagcx
export FLAGCX_PATH=/sgl-workspace/FlagCX

MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=2 \
  python examples/qwen3_6_27b_mtp_inference.py \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-overlap-schedule

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=2 \
  python examples/qwen3_6_27b_mtp_inference.py \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-overlap-schedule
```

Both commands select one node, TP2/PP1, BF16 model/KV, HCU attention and
synchronous eager execution. They use `EAGLE` with three steps, topk=1 and four
draft tokens from the same checkpoint; this is distinct from `EAGLE3`.
The profile fixes `page_size=64`, `context_length=4096`,
`mem_fraction_static=0.70`, `max_running_requests=16` and
`chunked_prefill_size=4096`, with radix caching, both graph phases and overlap off.
Each Hygon engine construction reasserts vendor entry-point isolation for its
spawned workers. Keep the environment setting before starting Python as shown.

MTP graph, branching topk>1 trees and overlap are explicitly rejected by the
current implementation. Engine B4 batched text is validated. Other topologies
and MTP vision/C64/HTTP paths remain outside its validated scope. These are software and validation boundaries.
Other platforms' defaults and ordinary non-MTP graph use are unchanged.

Keep the baseline enabled for full acceptance. All semantic contracts, the
64-token teacher-forced comparison, actual speculative counters and every
scheduler's acceptance length strictly greater than two must pass, with raw
exit and cleanup evidence retained. Missing head weights or acceptance data
fail. Component tests and single-request token/s are not full model acceptance
or controlled performance results.
### Hardware validation (2026-10-23)

| Check | Qwen3.6-27B | Qwen3.6-35B-A3B |
| --- | --- | --- |
| Full MTP/baseline example | 30 calls, 24 semantic checks passed | 30 calls, 24 semantic checks passed |
| Teacher-force comparison | 64 tokens; max logprob delta 0.242519; 0 near ties, 0 violations | 64 tokens; max delta 0.331164; 1 near tie, 0 violations |
| Full example scheduler acceptance window | 2.9916667 | 2.8444444 |
| Batched MTP | 4 batches of 4: all 16 semantic/token/logprob/top5/counter checks passed | Same 16 checks passed |
| Separate statistics probe | 256 actual output tokens, 91 verifies | 256 actual output tokens, 100 verifies |

The separate probes are excluded from the 16 batch requests. Original numerical
and acceptance gates are unchanged. All 236 Hygon unit tests and 19 actual GPU
numerical tests passed with no skips. Source hashes, responses, raw driver and
docker-exec exits (all zero), and cleanup were independently audited. Assigned
GPUs were at about 2 MiB before and after the intentional container stop;
scheduler diagnostics and resource-tracker warnings remain in the evidence.
This does not establish natural exit of every rank or absence of IPC leaks.

Every generating call had its direct output IDs and lengths checked; numerical
rescoring covered 64 teacher-forced tokens per model. These results establish
correctness for the stated profile, without a controlled MTP speedup claim.
Latest-head GitHub hardware CI, including the new eager MTP group, is reported
separately in the PR.
