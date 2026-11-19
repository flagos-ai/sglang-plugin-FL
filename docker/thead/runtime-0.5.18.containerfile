# syntax=docker/dockerfile:1.7
# SGLang 0.5.18 two-wheel CI image on the T-Head PPU operations runtime.
# Assembly only; build and evidence are recorded in docker/thead/README.md.
# Moving semver tag per the Runtime owner; the build records the resolved digest.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0
FROM ${BASE_IMAGE}

# The Runtime's own package index (its PIP_INDEX_URL) serves the SGLang closure.
ARG RUNTIME_INDEX=https://mirrors.cloud.tencent.com/pypi/simple

# One uv resolver install. The constraints, exclusion list and wheels are
# read-only bind mounts, so they never enter an image layer; no standalone
# triton is installed (the Runtime's /opt/flagtree tree is the compiler). The
# Runtime's Bash environment (BASH_ENV) exposes PYTHONPATH; the plugin
# activation variables live with the test configuration
# (tests/platforms/thead.yaml, env_defaults).
RUN --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --default-index "$RUNTIME_INDEX" \
      --constraints /tmp/artifacts/runtime-constraints.txt \
      --excludes /tmp/artifacts/runtime-0.5.18.excludes.txt \
      '/tmp/artifacts/sglang-0.5.18-py3-none-any.whl[srt_empty,runtime_common]' \
      '/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[thead]'
