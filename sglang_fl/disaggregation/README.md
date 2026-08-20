# FlagCX PD Disaggregation (xPyD) Guide

FlagCX's P2P engine unifies memory registration and transfer across multiple chip backends, so the
same connector works on every supported vendor. Before using the FlagCX connector you must clone and
build FlagCX yourself.

Source layout:

| File | Role |
|---|---|
| `patch.py` | Registers `flagcx` as a valid `--disaggregation-transfer-backend` value, injects the `TransferBackend` enum member, and wraps `get_kv_class` |
| `conn.py` | `FlagcxKVManager` / `FlagcxKVSender` / `FlagcxKVReceiver` / `FlagcxKVBootstrapServer` |
| `transfer_engine.py` | `FlagCXTransferEngine` — memory registration and batched one-sided writes via `flagcxP2pBatchWriteSync` |

The plugin is auto-loaded through its setuptools entry point, and Layer 4 is enabled by default
(`SGLANG_FL_DISAGG_FLAGCX=1`, see `sglang_fl/__init__.py:812`). The FlagCX shared library is loaded
**lazily**: `conn.py` is only imported once you actually pass
`--disaggregation-transfer-backend flagcx`.

---

## Supported Models

| Model | Quantization | NVIDIA | PPU |
|---|---|---|---|
| Qwen3.6-35B-A3B | BF16 | ✅ | ✅ |
| GLM-5.2-W4A8 | W4A8 | ✅ | ✅ |

---

## Prerequisites

1. Follow [sglang-plugin-FL/README.md](https://github.com/flagos-ai/sglang-plugin-FL/blob/main/README.md)
   to install the required packages. Note: `sglang[all]>=0.5.11` is required.

2. Make sure FlagCX builds cleanly:

   ```bash
   git clone https://github.com/flagos-ai/FlagCX.git
   cd FlagCX && make USE_NVIDIA=1
   ```

   If the build fails on your platform, see
   https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md

3. **Install `sglang-router`**: `pip install sglang-router`.

---

## Environment Variables

Set these on both roles (prefill / decode):

```bash
export SGLANG_FL_OOT_ENABLED=0        # recommended off when only exercising FlagCX PD disaggregation
export USE_FLAGGEMS=0                 # recommended off when only exercising FlagCX PD disaggregation
export FLAGCX_PATH=/path/to/FlagCX
export FLAGCX_P2P_SLICE_SIZE=65536    # slice size per one-sided RDMA write (recommended)
export FLAGCX_P2P_WORKERS_PER_POOL=2  # transfer workers per pool (recommended)
export FLAGCX_P2P_QPS_PER_CONN=2      # queue pairs per connection (recommended)
export SGLANG_SEND_AUX_TCP=0          # 1 → send aux data (output ids/logprobs) over TCP instead of RDMA
```

| Variable | Description |
|---|---|
| `FLAGCX_PATH` | FlagCX installation directory — **required** |
| `FLAGCX_P2P_SLICE_SIZE` | Bytes per one-sided write slice; raise it for long sequences |
| `FLAGCX_P2P_WORKERS_PER_POOL` | Number of transfer-pool workers |
| `FLAGCX_P2P_QPS_PER_CONN` | Queue pairs per connection; raise it to get more concurrency across multiple NICs |
| `FLAGCX_IB_HCA` | Usually **no need to set manually**: `conn.py:257` applies `setdefault` using the value of `--disaggregation-ib-device` |
| `SGLANG_FL_DISAGG_FLAGCX` | Layer 4 switch, default `1`; setting `0` makes `flagcx` an invalid backend value |
| `SGLANG_FL_OOT_ENABLED` / `USE_FLAGGEMS` | Layer 2 / Layer 1 switches; enable them one at a time during performance tuning |
| `SGLANG_SEND_AUX_TCP` | Optional, default:0; Send aux data (first-token metadata) over TCP instead of RDMA |

Throughout this guide, replace the placeholders with your own values:

| Placeholder | Meaning |
|---|---|
| `<MODEL_PATH>` | Local path to the model weights |
| `<PREFILL_HOST>` | IP or hostname of the prefill node |
| `<DECODE_HOST>` | IP or hostname of the decode node |
| `<ROUTER_HOST>` | IP or hostname of the node running the router |

---

## Step 1 — Launch prefill on node 1

```bash
mkdir -p log

export SGLANG_FL_OOT_ENABLED=0
export USE_FLAGGEMS=0
export FLAGCX_PATH=/path/to/FlagCX
export FLAGCX_P2P_SLICE_SIZE=65536
export FLAGCX_P2P_WORKERS_PER_POOL=2
export FLAGCX_P2P_QPS_PER_CONN=2

python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 8 \
    --max-running-requests 256 \
    --mem-fraction-static 0.85 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend flagcx \
    --disaggregation-ib-device mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8 \
    > log/prefill.log 2>&1 &
```

## Step 2 — Launch decode on node 2

Only two things differ from prefill: `--disaggregation-mode decode` and the log filename.
**`--disaggregation-transfer-backend` must be `flagcx` on both sides.**

```bash
mkdir -p log

export SGLANG_FL_OOT_ENABLED=0
export USE_FLAGGEMS=0
export FLAGCX_PATH=/path/to/FlagCX
export FLAGCX_P2P_SLICE_SIZE=65536
export FLAGCX_P2P_WORKERS_PER_POOL=2
export FLAGCX_P2P_QPS_PER_CONN=2

python -m sglang.launch_server \
    --model-path <MODEL_PATH> \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 8 \
    --max-running-requests 256 \
    --mem-fraction-static 0.85 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend flagcx \
    --disaggregation-ib-device mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8 \
    > log/decode.log 2>&1 &
```

Both logs must show `Layer 4 (PD Disagg): flagcx` in the plugin banner, plus:

```
FlagCX PD disaggregation backend registered (--disaggregation-transfer-backend flagcx)
```

If it is missing, the patch did not take effect (`sglang_fl/disaggregation/patch.py:139`), and SGLang
will exit with an error because `flagcx` is not a valid choice.

## Step 3 — Launch the router

`--mini-lb` is the built-in lightweight load balancer, which is enough for a 1P1D setup.

```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://<PREFILL_HOST>:8000 \
    --decode  http://<DECODE_HOST>:8000 \
    --mini-lb \
    --port 30000 \
    > router.log 2>&1 &
```

**Scaling to xPyD**: pass `--prefill` / `--decode` once per instance:

```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://<PREFILL_HOST_1>:8000 \
    --prefill http://<PREFILL_HOST_2>:8000 \
    --decode  http://<DECODE_HOST_1>:8000 \
    --decode  http://<DECODE_HOST_2>:8000 \
    --decode  http://<DECODE_HOST_3>:8000 \
    --policy round_robin \
    --port 30000 \
    > router.log 2>&1 &
```

> A single decode instance can usually absorb the output of several prefill instances. Pick the
> actual x:y ratio from benchmark data — prefill queueing time versus decode memory usage — rather
> than guessing from defaults.

## Step 4 — Single-request check

Send requests to the **router port (30000)**, not to the servers' port 8000.

```bash
curl -v http://<ROUTER_HOST>:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "base_model",
    "messages": [
      {"role": "user", "content": "What is the capital of China? And the capital of the United States?"}
    ],
    "temperature": 0,
    "max_tokens": 320
  }'
```

A valid response means the whole path works: bootstrap handshake, one-sided KV write, and decode
token generation.

## Step 5 — Multi-request benchmark

```bash
#!/bin/bash
# Disable Hugging Face network access on air-gapped machines; the tokenizer is read from a local path
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH=<MODEL_PATH>
BASE_ARGS="python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url http://<ROUTER_HOST>:30000 \
  --model $MODEL_PATH \
  --served-model-name base_model \
  --tokenizer $MODEL_PATH \
  --dataset-name random \
  --random-range-ratio 1.0 \
  --warmup-requests 20"

run_test() {
  local input_len=$1
  local output_len=$2
  local rps=$3
  local num_prompts=$(( rps * 5 ))
  echo "========================================"
  echo "Testing input=${input_len} output=${output_len} rps=${rps} num_prompts=${num_prompts}"
  echo "========================================"
  $BASE_ARGS \
    --random-input-len  "$input_len"  \
    --random-output-len "$output_len" \
    --request-rate      "$rps"        \
    --num-prompts       "$num_prompts"
  # Flush the prefix cache between rounds so hits from the previous round do not skew TTFT
  curl -X POST http://<ROUTER_HOST>:30000/flush_cache
}

# input   output  rps
run_test  1024    1024   75
run_test  4096    1024   70
run_test  8192    1024   65
run_test  16384   1024   60
run_test  32768   1024   45

echo "All tests done."
```

```bash
bash bench.sh > log/bench.log 2>&1 &
```

---

## End-to-end Path

```
                      ┌──────────────────────────┐
   client ───────────►│  sglang_router (:30000)  │
                      │  --pd-disaggregation     │
                      │  --mini-lb               │
                      └───────┬──────────┬───────┘
                              │          │
                 ① forward prompt        │② forward decode request
                              ▼          ▼
      ┌───────────────────────────┐   ┌───────────────────────────┐
      │ Node1  prefill (:8000)    │   │ Node2  decode (:8000)     │
      │ --disaggregation-mode     │   │ --disaggregation-mode     │
      │       prefill             │   │       decode              │
      │ --transfer-backend flagcx │   │ --transfer-backend flagcx │
      ├───────────────────────────┤   ├───────────────────────────┤
      │ FlagcxKVBootstrapServer   │◄──┤ FlagcxKVReceiver          │
      │   (ZMQ handshake)         │ ③ │   send_metadata / register │
      ├───────────────────────────┤   ├───────────────────────────┤
      │ FlagcxKVSender            │   │                           │
      │  → FlagcxKVManager        │   │                           │
      │     transfer_worker pool  │   │                           │
      │  → FlagCXTransferEngine   │   │                           │
      └────────────┬──────────────┘   └────────────▲──────────────┘
                   │  ④ flagcxP2pBatchWriteSync    │
                   │     one-sided RDMA write of   │
                   │     the KV cache              │
                   └───────────────────────────────┘
                        mlx5_bond_1..8 (FLAGCX_IB_HCA)

  Plugin-side registration path (once, at process startup):
    entry_point sglang.srt.plugins → sglang_fl.load_plugin()
      └─ apply_disaggregation_patch()            patch.py:139
           ├─ _add_backend_choice()   adds "flagcx" to the CLI choices
           ├─ _add_enum_member()      injects TransferBackend.FLAGCX
           └─ _patch_get_kv_class()   flagcx → the four Flagcx* classes
                                      (conn.py is imported only here,
                                       so libflagcx.so stays lazy)
```
