#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Text inference on eight MTT S5000 GPUs with local BF16 model weights.
set -euo pipefail
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DIST_BACKEND=flagcx
export SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero,cumsum,mm,unique,_unique2,unique_dim,unique_consecutive,max,topk
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=8
export SGLANG_FL_DISAGG_FLAGCX=0

python -m sglang.launch_server \
    --model-path "${MODEL_PATH:-/models/Qwen3.8-Flash-Next}" \
    --dtype bfloat16 --tp 8 \
    --mem-fraction-static 0.85 --chunked-prefill-size 16384 \
    --attention-backend fa3 \
    --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
    --mamba-ssm-dtype float32 --kv-cache-dtype bfloat16 \
    --qsa-indexer-dtype bfloat16 --sampling-backend pytorch \
    --reasoning-parser qwen3 --max-total-tokens 262144 \
    --max-running-requests 64 --disable-radix-cache --disable-overlap-schedule \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
    --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
    --cuda-graph-backend-prefill disabled --skip-server-warmup \
    --host 0.0.0.0 --port "${PORT:-30000}"
