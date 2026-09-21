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
