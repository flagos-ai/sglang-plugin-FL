# SGLang 0.5.18 empty-mode image for sglang-plugin-FL on T-Head PPU (PPU-ZW810E).
#
# Built on the operations-owned T-Head PPU runtime.  The immutable runtime carries
# PPU SDK 2.1.0, vendor Torch 2.10.0, FlagTree 0.7.0+ppu3.6, FlagGems 5.4.0 and the
# matching vendor torchvision/torchaudio wheels.  This image must preserve those
# components and add only SGLang 0.5.18 srt_empty, its application requirements and
# sglang-plugin-FL.  Platform-specific components that are absent from the runtime
# must be treated as explicit validation findings; do not silently inherit them from
# a development/release image.
#
# Build from the repository root:
#   docker build -f docker/thead/empty-0.5.18.containerfile \
#     -t sglang-plugin-fl:thead-empty-0.5.18 .
# BASE_IMAGE is the runtime rebuilt by build-infra PR #1738.  That rebuild adds
# torchvision 0.25.0+v0.1.0.ppu2.1.1 and torchaudio
# 2.10.0+v0.1.0.ppu2.1.1; see build-infra issue #1730.  Keep it digest-pinned.
# If the default PIP_INDEX_URL (official PyPI) is slow from the build host, pass
# --build-arg PIP_INDEX_URL=https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu130/simple/
#
# Run: PPU devices are passed with --device /dev/alixpu* (see .github/configs/thead.yml).
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0@sha256:163faf04cac90612954f32b0f45eef90223708fd3de0408497570ee4fd3b408d
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
#   Do not set SGLANG_FL_FLAGOS_BLACKLIST: it replaces the blacklist in
#     sglang_fl/dispatch/config/thead.yaml instead of extending it.
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false

# CI runs offline against the mounted model store (/data/models/Qwen, see
# .github/configs/thead.yml). HF_HOME is pinned because the runner overrides
# HOME=/github/home at job time. Same settings as docker/thead/containerfile.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /opt/sglang-plugin-fl
