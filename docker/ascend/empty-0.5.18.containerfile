# Upgrade the validated 910C empty runtime to SGLang 0.5.18 while preserving
# its CANN 8.5 / torch 2.8 / torch_npu 2.8 vendor stack.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-ascend-empty-tree_none-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.11.14-torch_npu_2.8.0.post2-pcp_cann8.5.0-gpu_a3-arc_arm64-driver_25.5.0:202608291915
FROM ${BASE_IMAGE} AS runtime

ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
ARG SGLANG_VERSION=0.5.18
ARG FLAGGEMS_COMMIT=98fae44cdf2898f39c7f24f080d7c88b83d7c593
ARG FLAGGEMS_VERSION=5.3.0
ARG FLAGCX_COMMIT=68f069fe4ff2af9e8017b74aee8dee60c59e3b1d
ARG SGL_KERNEL_NPU_RELEASE=2026.05.01.post2
ARG SGL_KERNEL_NPU_VERSION=2026.5.1
ARG SGL_KERNEL_NPU_SHA256=1c446e04b23497b97089591713637d512ed16135d0d97d71b6cf5c7db6787fc5
ARG SGL_KERNEL_NPU_ARCHIVE=sgl-kernel-npu-2026.05.01.post2-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip

ENV SGLANG_BUILD_RUST_EXTS=none \
    FLAGCX_PATH=/opt/FlagCX \
    PATH=/usr/local/Ascend/driver/tools:${PATH}

RUN command -v curl >/dev/null && \
    command -v git >/dev/null && \
    command -v make >/dev/null && \
    command -v unzip >/dev/null

# Install the official SGLang v0.5.18 source through its torch-free empty extra.
# Constraints are defensive: no transitive dependency may replace the vendor
# torch, torch_npu, or Ascend Triton packages supplied by the base image.
RUN mkdir -p /opt/sglang-0.5.18 && \
    python3 -m pip uninstall -y sglang && \
    rm -rf /sgl-workspace/sglang && \
    curl --fail --location --retry 5 --retry-delay 5 \
        "https://codeload.github.com/sgl-project/sglang/tar.gz/${SGLANG_COMMIT}" \
        --output /tmp/sglang.tar.gz && \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /opt/sglang-0.5.18 && \
    printf '%s\n' "${SGLANG_COMMIT}" > /opt/sglang-0.5.18/.flagos-source-commit && \
    cd /opt/sglang-0.5.18/python && \
    cp pyproject_other.toml pyproject.toml && \
    printf '%s\n' \
        'torch==2.8.0' \
        'torch-npu==2.8.0.post2' \
        'triton==3.5.0' \
        'triton-ascend==3.2.0' \
        > /tmp/ascend-constraints.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
        python3 -m pip install --no-cache-dir \
        -c /tmp/ascend-constraints.txt '.[srt_empty]' && \
    python3 -m pip install --no-cache-dir --no-deps \
        xgrammar==0.2.1 compressed-tensors==0.15.0 && \
    rm -f /tmp/sglang.tar.gz

# Install the exact FlagGems source used by this Ascend adaptation. Build tools
# are pinned and dependency resolution is disabled so the vendor runtime stays
# unchanged.
RUN python3 -m pip install --no-cache-dir \
        'setuptools>=68,<77' \
        'setuptools-scm>=8,<10' \
        scikit-build-core==0.12.2 \
        pybind11==3.0.3 \
        'cmake>=3.20,<4.0' \
        ninja==1.13.0 \
        wheel && \
    mkdir -p /opt/FlagGems && \
    curl --fail --location --retry 5 --retry-delay 5 \
        "https://codeload.github.com/flagos-ai/FlagGems/tar.gz/${FLAGGEMS_COMMIT}" \
        --output /tmp/flaggems.tar.gz && \
    tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems && \
    printf '%s\n' "${FLAGGEMS_COMMIT}" > /opt/FlagGems/.flagos-source-commit && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=${FLAGGEMS_VERSION} \
        python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        -e /opt/FlagGems && \
    rm -f /tmp/flaggems.tar.gz

# Keep the real CANN 8.5 / torch 2.8 NPU kernel wheel. SGLang 0.5.18 modules
# are checked explicitly below; unsupported kernel surfaces fail the image build
# instead of being hidden by an import shim.
RUN curl --fail --location --retry 5 --retry-delay 5 \
        "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGL_KERNEL_NPU_RELEASE}/${SGL_KERNEL_NPU_ARCHIVE}" \
        --output "/tmp/${SGL_KERNEL_NPU_ARCHIVE}" && \
    echo "${SGL_KERNEL_NPU_SHA256}  /tmp/${SGL_KERNEL_NPU_ARCHIVE}" | sha256sum --check - && \
    mkdir -p /tmp/sgl-kernel-npu-wheels && \
    unzip -j "/tmp/${SGL_KERNEL_NPU_ARCHIVE}" '*.whl' -d /tmp/sgl-kernel-npu-wheels && \
    test -f /tmp/sgl-kernel-npu-wheels/attentions-0.2-cp311-cp311-linux_aarch64.whl && \
    test -f /tmp/sgl-kernel-npu-wheels/torch_memory_saver-0.0.8-cp311-cp311-linux_aarch64.whl && \
    test -f "/tmp/sgl-kernel-npu-wheels/sgl_kernel_npu-${SGL_KERNEL_NPU_VERSION}-cp311-cp311-linux_aarch64.whl" && \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        /tmp/sgl-kernel-npu-wheels/attentions-0.2-cp311-cp311-linux_aarch64.whl \
        /tmp/sgl-kernel-npu-wheels/torch_memory_saver-0.0.8-cp311-cp311-linux_aarch64.whl \
        "/tmp/sgl-kernel-npu-wheels/sgl_kernel_npu-${SGL_KERNEL_NPU_VERSION}-cp311-cp311-linux_aarch64.whl" && \
    rm -rf "/tmp/${SGL_KERNEL_NPU_ARCHIVE}" /tmp/sgl-kernel-npu-wheels

# Build FlagCX at the exact post-v0.13.0 revision required by the 910C setup.
# The optional BuildKit secret contains only the git proxy URL and is never
# copied into an image layer.
RUN --mount=type=secret,id=git_proxy,required=false \
    set -eu; \
    rm -rf /opt/FlagCX; \
    git init -q /opt/FlagCX; \
    git -C /opt/FlagCX remote add origin https://github.com/flagos-ai/FlagCX.git; \
    if [ -s /run/secrets/git_proxy ]; then \
        proxy="$(cat /run/secrets/git_proxy)"; \
        git -C /opt/FlagCX -c http.proxy="${proxy}" fetch --depth 1 origin "${FLAGCX_COMMIT}"; \
        git -C /opt/FlagCX checkout -q --detach FETCH_HEAD; \
        git -C /opt/FlagCX -c http.proxy="${proxy}" submodule update --init --recursive; \
    else \
        git -C /opt/FlagCX fetch --depth 1 origin "${FLAGCX_COMMIT}"; \
        git -C /opt/FlagCX checkout -q --detach FETCH_HEAD; \
        git -C /opt/FlagCX submodule update --init --recursive; \
    fi; \
    test "$(git -C /opt/FlagCX rev-parse HEAD)" = "${FLAGCX_COMMIT}"; \
    printf '%s\n' "${FLAGCX_COMMIT}" > /opt/FlagCX/.flagos-source-commit; \
    toolkit_home="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"; \
    test -d "${toolkit_home}"; \
    make -C /opt/FlagCX USE_ASCEND=1 CCL_HOME="${toolkit_home}" -j"$(nproc)"; \
    test -f /opt/FlagCX/build/lib/libflagcx.so; \
    test -f /opt/FlagCX/plugin/interservice/flagcx_wrapper.py; \
    rm -rf /opt/FlagCX/.git

# Bake the matching plugin into the deployable runtime. CI later overlays the
# checked-out source with an editable --no-deps install.
COPY pyproject.toml /opt/sglang-plugin-fl/pyproject.toml
COPY sglang_fl /opt/sglang-plugin-fl/sglang_fl
RUN python3 -m pip uninstall -y sglang-fl >/dev/null 2>&1 || true && \
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        /opt/sglang-plugin-fl

COPY .github/scripts/ascend/verify_environment.py /usr/local/bin/verify-sglang-fl-ascend
RUN chmod 0755 /usr/local/bin/verify-sglang-fl-ascend && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        python3 /usr/local/bin/verify-sglang-fl-ascend

LABEL ai.flagos.sglang.version="${SGLANG_VERSION}" \
      ai.flagos.sglang.revision="${SGLANG_COMMIT}" \
      ai.flagos.flaggems.revision="${FLAGGEMS_COMMIT}" \
      ai.flagos.flagcx.revision="${FLAGCX_COMMIT}" \
      ai.flagos.cann.version="8.5.0" \
      ai.flagos.torch-npu.version="2.8.0.post2"

FROM runtime AS ci

RUN python3 -m pip install --no-cache-dir \
        pytest==9.1.1 \
        pytest-timeout==2.4.0 \
        pytest-cov \
        pytest-json-report \
        pytest-metadata \
        pyyaml \
        ruff && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        python3 /usr/local/bin/verify-sglang-fl-ascend --require-ci

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace

# Build from the repository root so the plugin and verifier are in context:
#   docker build --platform linux/arm64 --target ci \
#     --secret id=git_proxy,env=GIT_PROXY \
#     -f docker/ascend/empty-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.2.0-ascend-sglang0.5.18-ci .
# Push that tag first, then replace the CI reference with its registry digest.
