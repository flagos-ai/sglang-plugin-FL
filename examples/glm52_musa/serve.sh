#!/usr/bin/env bash
# Run inside the 2026-10-27 team SGLang/MUSA image on each node.
set -euo pipefail

rank="${1:?node rank 0 or 1 is required}"
case "$rank" in 0|1) ;; *) echo 'node rank must be 0 or 1' >&2; exit 2 ;; esac
export GLM52_MODEL_PATH="${GLM52_MODEL_PATH:?model directory is required}"
: "${GLM52_DIST_ADDR:?node 0 address and distributed port are required}"
: "${GLM52_DEEPEP_CONFIG:?DeepEP config path is required}"
: "${GLM52_SOCKET_IFNAME:?cluster network interface is required}"
: "${GLM52_IB_DEVICES:?comma-separated HCA names are required}"
: "${GLM52_IB_GID_INDEX:?RDMA GID index is required}"

export PYTHONUNBUFFERED=1
# Preserve all SGLang environment settings inherited from the original image.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_PADDING=1
export SGLANG_USE_DEEPGEMM_BMM=0
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_MUSA_FUSED_KERNEL=1
# Compatibility: avoid same-host cross-rail GIN QP timeout; normal ACE remains enabled.
export MCCL_GIN_ENABLE=0
# Restore original-image default FP8 DeepGEMM execution path.
export SGLANG_ENABLE_JIT_DEEPGEMM=1
# Preserve the old-image FP8 indexer precision; new image defaults to unsupported INT8.
export MX_ENABLE_NSA_INT8=0
export SGLANG_DSV4_INDEX_CACHE_TYPE=fp8
export GLM52_MATE_DSA_NO_SPLIT=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:False
source "$(dirname "${BASH_SOURCE[0]}")/operator-profile.sh"
export MCCL_SOCKET_IFNAME="$GLM52_SOCKET_IFNAME"
export NCCL_SOCKET_IFNAME="$GLM52_SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$GLM52_SOCKET_IFNAME"
export TP_SOCKET_IFNAME="$GLM52_SOCKET_IFNAME"
export MCCL_IB_DISABLE=0
export NCCL_IB_DISABLE=0
export MCCL_IB_GID_INDEX="$GLM52_IB_GID_INDEX"
export NCCL_IB_GID_INDEX="$GLM52_IB_GID_INDEX"
export SGLANG_IB_DEVICES="$GLM52_IB_DEVICES"
export MCCL_IB_HCA="$SGLANG_IB_DEVICES"
export NCCL_IB_HCA="$SGLANG_IB_DEVICES"
export SGLANG_USE_MTT_INDEXER=0
export SGLANG_USE_MTT_ATTN=0
export SGLANG_USE_MTT_F8GEMM=0
export SGLANG_USE_MTT_HGEMM=0
export SGLANG_USE_MTT_EPI_QKV=0
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_PP_LAYER_PARTITION=40,38
export SGLANG_OPT_USE_TRITON_MASKED_SWIGLU_QUANT=1
export SGLANG_DEEP_GEMM_BLOCK_M=256
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export SGLANG_FUSED_MLA_ROPE_CACHE=1
export MCCL_PROTOS=2
export MCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_LEVEL=5
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_RING_16P1H=0
export MCCL_TOPO_ENHANCE=0
export MCCL_NIC_ENHANCE=0
export MCCL_MULTINODE_ENHANCE=0
export MCCL_GRAPH_HELPER_DISABLE=1
export SGLANG_DEEPEP_USE_MUSA_ACE=1
export SGLANG_SBO_COMBINE_SHARED=1
export SGLANG_DEEPEP_BF16_DISPATCH=0
export DEEPEP_CPU_POLL_TIMEOUT=200

exec python3 -u -m sglang.launch_server \
  --model-path "$GLM52_MODEL_PATH" \
  --served-model-name GLM-5.2 \
  --trust-remote-code \
  --host 0.0.0.0 --port "${GLM52_PORT:-24587}" \
  --disable-overlap-schedule \
  --enable-single-batch-overlap \
  --cuda-graph-max-bs 32 --cuda-graph-bs 8 16 24 32 \
  --dp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --tp-size 8 --ep-size 8 --pp-size 2 \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --fp8-gemm-backend auto \
  --attention-backend nsa \
  --nsa-prefill-backend trtllm \
  --nsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-config "$GLM52_DEEPEP_CONFIG" \
  --sampling-backend flashinfer \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 147456 \
  --context-length 147456 \
  --dist-init-addr "$GLM52_DIST_ADDR" \
  --nnodes 2 --node-rank "$rank" \
  --tokenizer-backend fastokens \
  --load-balance-method round_robin \
  --watchdog-timeout 900 \
  --log-level info \
  --enable-metrics \
  --enable-cache-report
