#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Single-node MTT S5000 TP8, dynamic symmetric token/channel INT8.
# Requires vendor Torch-MUSA 2.9 / MUSA 4.3 and FlagTree 0.7.0 (Triton 3.6).
# SGLang v0.5.18: 71de97b264b04dcd514cf904003028aefe9775c8.
# FlagGems: 97dbf2fd19cb9d81d5115399f7e765be5f7a7ec6.
# Private upstream selection hooks are version-bound; do not upgrade without
# repeating installation, numerical, model and concurrent-request validation.
# Supply a local compressed-tensors MiniMax-M3.1-Flash W8A8 checkpoint.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-M3.1-Flash W8A8 checkpoint}"
export USE_FLAGGEMS=1
# FlagTree automatic tile shrinking violates sparse-topk static constraints.
export FLAGTREE_AABS=0
# Use the existing full-FP32 router; MUSA lacks aten::mm.dtype.
export SGLANG_OPT_USE_BF16_ROUTER_GEMM=0
export SGLANG_FL_DIST_BACKEND="${SGLANG_FL_DIST_BACKEND:-mccl}"
export SGLANG_FLAGGEMS_RECORD=1
export SGLANG_FLAGGEMS_LOG_PATH="${SGLANG_FLAGGEMS_LOG_PATH:-/tmp/minimax-m31-flaggems.log}"
export SGLANG_FL_DISPATCH_DEBUG=1
export SGLANG_FL_DISPATCH_LOG="${SGLANG_FL_DISPATCH_LOG:-/tmp/minimax-m31-dispatch.log}"

exec "${PYTHON_BIN:-python3}" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name MiniMax-M3.1-Flash-W8A8 \
    --trust-remote-code \
    --dtype bfloat16 \
    --device musa \
    --tp-size 8 \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-30000}" \
    --context-length "${CONTEXT_LENGTH:-81920}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-8}" \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 4096 \
    --prefill-max-requests 1 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --reasoning-parser minimax-m3 \
    --limit-mm-data-per-request '{"image":16}' \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --disable-radix-cache \
    --enable-metrics \
    "$@"
