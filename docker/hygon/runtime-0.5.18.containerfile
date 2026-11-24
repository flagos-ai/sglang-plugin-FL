# syntax=docker/dockerfile:1.7
# SGLang 0.5.18 two-wheel CI image on the Hygon operations runtime: assembly
# only. Build, acceptance, evidence and removal gates: docker/hygon/README.md
# Moving semver tag per the Runtime owner; the build records the resolved digest.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04:2.3.0
FROM ${BASE_IMAGE}

# TEMP (FlagOS Runtime): /opt/flagtree is exposed to bash sessions only; also the
# Triton compiler slot. Evidence/owner/removal gate: docker/hygon/README.md
ENV PYTHONPATH=/opt/flagtree

# Index strategy unsafe-first-match: first-index stops at the Hygon index for
# a name it carries without a matching version (evidence: docker/hygon/README.md).
ARG FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
ARG TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple
ARG PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

# One uv resolver install: constraints, excludes and wheels are read-only bind
# mounts that never enter an image layer, and no standalone triton is installed
# (the base /opt/flagtree tree is the compiler). Evidence: docker/hygon/README.md
RUN --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --index "$FLAGOS_INDEX" \
      --index "$TENCENT_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
      --constraints /tmp/artifacts/runtime-constraints.txt \
      --excludes /tmp/artifacts/runtime-0.5.18.excludes.txt \
      '/tmp/artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
      '/tmp/artifacts/sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[hygon]'

# --- TEMP Runtime compatibility bundle ---------------------------------------
# TEMP Runtime compatibility bundle；这些设置暴露 Runtime 已携带但没有对所有入口
# 稳定提供的能力；等待 Runtime owner 明确迁移方案；详细 observed failure、owner
# 和 removal gate 见 docker/hygon/README.md。
# 1/3 DTK loader path, prepended to whatever the base carries.
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# 2/3 FlagTree derives its clang selection from this root.
ENV ROCM_PATH=/opt/dtk
# 3/3 The ROCm prefix alias SGLang's HIP JIT builder expects.
RUN if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then ln -s /opt/dtk /opt/rocm; fi \
    && test "$(readlink -f /opt/rocm)" = "$(readlink -f /opt/dtk)"
