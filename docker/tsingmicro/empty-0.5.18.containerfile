# TsingMicro TX8110 (TXDA) SGLang 0.5.18 empty runtime.
#
# Single Python environment: the system Python carries SGLang 0.5.18 [srt_empty]
# and this plugin; SGLang 0.5.11, the old plugin copy and every CUDA-only wheel
# are removed. The TXDA vendor stack of the base image (torch 2.11.0+cpu,
# torch_txda, FlagTree 0.7.1a1+tsingmicro3.6, FlagGems, Kuiper runtime/CCL
# 0.30279.6, FlagCX) is preserved via a captured constraints file.
#
# Build from the repository root:
#   docker build -f docker/tsingmicro/empty-0.5.18.containerfile \
#       --target ci -t sglang-plugin-fl:0.5.18-tsingmicro-ci .
#   (--target runtime for the image without test tools)
#
# Offline / slow-GitHub sites: pass a mirror of the release tarball, e.g.
#   --build-arg SGLANG_SOURCE_URL=https://<mirror>/sglang-v0.5.18.tar.gz \
#   --build-arg SGLANG_SOURCE_SHA256=<sha256>
#
# Run (no --privileged; the TXDA device nodes are enough):
#   docker run --device /dev/accel --device /dev/accel_drv_mgr --ipc=host ...

# BASE_IMAGE: TsingMicro SGLang 0.5.11 image with the 0.30279.6 runtime/CCL.
# The default is the locally built slim image; replace it with the published
# Harbor tag/digest once available.
ARG BASE_IMAGE=sgl-qwen:20261109-slim-v4
FROM ${BASE_IMAGE} AS runtime

ARG SGLANG_VERSION=0.5.18
ARG SGLANG_SOURCE_URL
ARG SGLANG_SOURCE_SHA256
ARG PIP_INDEX_URL

COPY . /opt/sglang-plugin-fl
RUN if [ -n "${PIP_INDEX_URL}" ]; then export PIP_INDEX_URL; else unset PIP_INDEX_URL; fi && \
    export SGLANG_VERSION="${SGLANG_VERSION}" \
           SGLANG_SOURCE_URL="${SGLANG_SOURCE_URL:-https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/v${SGLANG_VERSION}}" \
           SGLANG_SOURCE_SHA256="${SGLANG_SOURCE_SHA256}" \
           PLUGIN_DIR=/opt/sglang-plugin-fl PLUGIN_EDITABLE=1 && \
    bash /opt/sglang-plugin-fl/docker/tsingmicro/install_sglang_0518.sh

# Validated TXDA defaults (also used by CI). Engine defaults (device, dtype,
# attention backends, ...) come from the plugin itself:
# sglang_fl/dispatch/backends/vendor/tsingmicro/runtime_config.py
ENV USE_FLAGGEMS=1 \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_FL_DIST_BACKEND=flagcx

WORKDIR /opt/sglang-plugin-fl

# CI stage: test tooling only; no dependency resolution happens in CI jobs, so
# pip cannot silently replace the vendor stack there.
FROM runtime AS ci
RUN python3 -m pip install --no-cache-dir -c /opt/tsingmicro-vendor-constraints.txt \
        'setuptools>=68,<82' wheel pytest pytest-cov pytest-timeout \
        pytest-json-report pytest-metadata pyyaml ruff && \
    python3 /opt/sglang-plugin-fl/docker/tsingmicro/verify_environment.py \
        --plugin-root /opt/sglang-plugin-fl --metadata-only
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /workspace
