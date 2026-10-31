# SGLang 0.5.18 empty-mode image for sglang-plugin-FL on T-Head PPU (PPU-ZW810E).
#
# Built on top of a T-Head PPU base that already carries the vendor stack (PPU SDK
# 2.1.0, PPU torch 2.10.0, FlagTree 0.7.0+ppu3.6, sgl-kernel ppu build, FlagGems,
# FlagCX under /sgl-workspace). Those builds are proprietary/vendor-built and are not
# reinstalled here. install-sglang-0.5.18-empty.sh pins them as pip constraints,
# installs SGLang 0.5.18 with the torch-free `srt_empty` extra, and installs the plugin.
# No SGLang source patch is needed: all PPU adaptation lives in the plugin.
#
# Build from the repository root:
#   docker build -f docker/thead/empty-0.5.18.containerfile \
#     -t sglang-plugin-fl:thead-empty-0.5.18 .
# BASE_IMAGE defaults to the PPU empty image the 0.5.11 adaptation was validated on
# (vendor stack above, SGLang 0.5.11 empty, Python 3.12.3); override it with
# --build-arg BASE_IMAGE=<image> only to build on another PPU empty base.
# If the default PIP_INDEX_URL (official PyPI) is slow from the build host, pass
# --build-arg PIP_INDEX_URL=https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu130/simple/
#
# Run: PPU devices are passed with --device /dev/alixpu* (see .github/configs/thead.yml).
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-zhenwu-empty-tree_0.7.0_ppu3.6-gems_5.5.0.dev0_g7536a18c0-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.3-torch_2.10.0-pcp_hggc13.0-gpu_ppu-zw810e-arc_amd64-driver_1.3.2-d7f5a2:202610191938
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

# CI runs offline against the mounted model store (/data/models/Qwen, see
# .github/configs/thead.yml). HF_HOME is pinned because the runner overrides
# HOME=/github/home at job time. Same settings as docker/thead/containerfile.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The base image adds /opt/accl-p (PPU pccl libraries) to LD_LIBRARY_PATH only in
# /root/.bashrc. GitHub Actions steps run `bash --noprofile --norc`, so bake it
# into the image env to match the validated runtime.
ENV LD_LIBRARY_PATH=/opt/accl-p:${LD_LIBRARY_PATH}

# Fail the build if the vendor FlagCX tree from the base image is incomplete.
RUN test -f "${FLAGCX_PATH}/build/lib/libflagcx.so" && \
    test -f "${FLAGCX_PATH}/plugin/interservice/flagcx_wrapper.py"

WORKDIR /opt/sglang-plugin-fl
