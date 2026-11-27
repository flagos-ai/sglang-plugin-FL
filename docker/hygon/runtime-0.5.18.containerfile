# syntax=docker/dockerfile:1.7
# SGLang 0.5.18 two-wheel CI image on the Hygon operations runtime: assembly
# only. Build, acceptance, evidence and removal gates: docker/hygon/README.md
# Moving semver tag per the Runtime owner; the build records the resolved digest.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04:2.3.0
FROM ${BASE_IMAGE}

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
