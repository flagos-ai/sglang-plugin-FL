# SGLang 0.5.18 empty-mode image for sglang-plugin-FL on T-Head PPU (PPU-ZW810E).
#
# Built on top of a T-Head PPU base that already carries the vendor stack (PPU SDK
# 2.1.0, PPU torch 2.10.0, FlagTree 0.7.0+ppu3.6, sgl-kernel ppu build, FlagGems,
# FlagCX under /sgl-workspace). Those builds are proprietary/vendor-built and are not
# reinstalled here. install-sglang-0.5.18-empty.sh pins them as pip constraints,
# installs SGLang 0.5.18 with the torch-free `srt_empty` extra, and installs the plugin.
# No SGLang source patch is needed: all PPU adaptation lives in the plugin.
#
# Build from the repository root (BASE_IMAGE is required; use the PPU empty image the
# 0.5.11 adaptation was validated on):
#   docker build -f docker/thead/empty-0.5.18.containerfile \
#     --build-arg BASE_IMAGE=<ppu-empty-base-image> \
#     -t sglang-plugin-fl:thead-empty-0.5.18 .
#
# Run: PPU devices are passed with --device /dev/alixpu* (see .github/configs/thead.yml).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION=0.5.18
# pip index: official PyPI, as in docker/thead/containerfile (the base image's
# aiext mirror is PPU/ML-focused). Override with --build-arg PIP_INDEX_URL=<url>.
# Longer timeout and retries because files.pythonhosted.org can be slow from the
# PPU build hosts.
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# A stale plugin copy in the base must not shadow the editable install below.
RUN pip uninstall -y sglang_fl 2>/dev/null || true

COPY . /opt/sglang-plugin-fl
# No PPU devices at build time: skip the runtime activation check.
RUN echo "pip index: ${PIP_INDEX_URL}" && \
    SGLANG_VERSION=${SGLANG_VERSION} SGLANG_SRC=/opt/sglang-${SGLANG_VERSION} \
    PLUGIN_SRC=/opt/sglang-plugin-fl SKIP_DEVICE_CHECK=1 \
    bash /opt/sglang-plugin-fl/docker/thead/install-sglang-0.5.18-empty.sh && \
    pip cache purge

# Test toolchain baked in so CI does not depend on PyPI at job time
# (.github/scripts/thead/setup.sh still re-installs the plugin under test).
RUN pip install --no-cache-dir pytest pytest-timeout pyyaml

# Runtime defaults for the empty runtime (match the validated 0.5.11 empty env).
#   SGLANG_IS_FLASHINFER_AVAILABLE=false: FlashInfer is not installed; the plugin also
#     serves an import placeholder (vendor/thead/import_guard.py) for module-level imports.
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1: the PPU sgl-kernel build (0.4.2.post2) is
#     older than the 0.4.6.post1 SGLang 0.5.18 pins for NVIDIA.
#   Do not set SGLANG_FL_FLAGOS_BLACKLIST: it replaces the blacklist in
#     sglang_fl/dispatch/config/thead.yaml instead of extending it.
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/sgl-workspace/FlagCX \
    SGLANG_IS_FLASHINFER_AVAILABLE=false \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

WORKDIR /opt/sglang-plugin-fl
