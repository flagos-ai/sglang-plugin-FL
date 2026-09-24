# Upgrade the validated 910C empty runtime to SGLang 0.5.18 while preserving
# its CANN 8.5 / torch 2.8 / torch_npu 2.8 vendor stack.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-ascend-empty-tree_none-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.11.14-torch_npu_2.8.0.post2-pcp_cann8.5.0-gpu_a3-arc_arm64-driver_25.5.0:202608291915
FROM ${BASE_IMAGE} AS runtime

ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
ARG SGLANG_VERSION=0.5.18
ARG SGLANG_FL_REVISION
ARG FLAGGEMS_COMMIT=98fae44cdf2898f39c7f24f080d7c88b83d7c593
ARG FLAGGEMS_VERSION=5.3.0
ARG FLAGCX_COMMIT=68f069fe4ff2af9e8017b74aee8dee60c59e3b1d
ARG SGL_KERNEL_NPU_RELEASE=2026.8.10
ARG SGL_KERNEL_NPU_COMMIT=e05fc90e85041d6080431c8ce543ab112c56e16d
ARG SGL_KERNEL_NPU_VERSION=2026.6.1
ARG SGL_KERNEL_NPU_SHA256=fa4ad5afa6bb748d683da5ee8a9cc702c13d0398d3db53651b4bb79e3f8d0e95
ARG SGL_KERNEL_NPU_SOLVE_TRIL_SHA256=d5aaeef310dae31c1cdfad0776c379a852b3e955c469b75cd53b783cfa192a37
ARG SGL_KERNEL_NPU_ARCHIVE=sgl-kernel-npu-2026.8.10-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip
ARG DEEP_EP_VERSION=1.0.0+e05fc90e.cann.8.5.0.b232

ENV SGLANG_BUILD_RUST_EXTS=none \
    FLAGCX_PATH=/opt/FlagCX \
    PATH=/usr/local/Ascend/driver/tools:${PATH}

RUN command -v curl >/dev/null && \
    command -v git >/dev/null && \
    command -v make >/dev/null && \
    command -v unzip >/dev/null

# The validated empty image deliberately combines the CPU-tagged PyTorch wheel
# with torch_npu. Fail before mutating the image if the vendor base drifts.
RUN python3 -c "import importlib.metadata as m,sys; expected={'torch':'2.8.0+cpu','triton-ascend':'3.2.1'}; actual={name:m.version(name) for name in expected}; sys.exit(f'expected {expected}, found {actual}') if actual != expected else print(actual)"

# Install the official SGLang v0.5.18 source through its torch-free empty extra.
# Constraints are defensive: no transitive dependency may replace the vendor
# torch, torch_npu, or Ascend Triton packages supplied by the base image.
RUN rm -rf /opt/sglang-0.5.18 && \
    mkdir -p /opt/sglang-0.5.18 && \
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
        'torch==2.8.0+cpu' \
        'torch-npu==2.8.0.post2' \
        'triton==3.5.0' \
        'triton-ascend==3.2.1' \
        > /tmp/ascend-constraints.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
        python3 -m pip install --no-cache-dir \
        -c /tmp/ascend-constraints.txt '.[srt_empty]' && \
    python3 -c "import importlib.metadata as m,sys; actual=m.version('sglang'); sys.exit(f'sglang must be ${SGLANG_VERSION}, found {actual}') if actual != '${SGLANG_VERSION}' else print('sglang', actual)" && \
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
    rm -rf /opt/FlagGems && \
    mkdir -p /opt/FlagGems && \
    curl --fail --location --retry 5 --retry-delay 5 \
        "https://codeload.github.com/flagos-ai/FlagGems/tar.gz/${FLAGGEMS_COMMIT}" \
        --output /tmp/flaggems.tar.gz && \
    tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems && \
    python3 -c "from pathlib import Path; import sys; p=Path('/opt/FlagGems/pyproject.toml'); s=p.read_text(); old='version = \"5.3.0rc2\"'; new='version = \"${FLAGGEMS_VERSION}\"'; count=s.count(old); sys.exit(f'expected one FlagGems version anchor, found {count}') if count != 1 else p.write_text(s.replace(old, new))" && \
    printf '%s\n' "${FLAGGEMS_COMMIT}" > /opt/FlagGems/.flagos-source-commit && \
    python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        -e /opt/FlagGems && \
    python3 -c "import importlib.metadata as m,sys; actual=m.version('flag-gems'); sys.exit(f'flag-gems must be ${FLAGGEMS_VERSION}, found {actual}') if actual != '${FLAGGEMS_VERSION}' else print('flag-gems', actual)" && \
    rm -f /tmp/flaggems.tar.gz

# Install the official SGLang 0.5.18 NPU kernel release, selecting its real
# CANN 8.5 / torch 2.8 / A3 / aarch64 build for this vendor runtime. The
# release tag intentionally differs from the wheel version embedded in it.
RUN curl --fail --location --retry 5 --retry-delay 5 \
        "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGL_KERNEL_NPU_RELEASE}/${SGL_KERNEL_NPU_ARCHIVE}" \
        --output "/tmp/${SGL_KERNEL_NPU_ARCHIVE}" && \
    echo "${SGL_KERNEL_NPU_SHA256}  /tmp/${SGL_KERNEL_NPU_ARCHIVE}" | sha256sum --check - && \
    mkdir -p /tmp/sgl-kernel-npu-wheels && \
    unzip -j "/tmp/${SGL_KERNEL_NPU_ARCHIVE}" '*.whl' -d /tmp/sgl-kernel-npu-wheels && \
    test -f "/tmp/sgl-kernel-npu-wheels/deep_ep-${DEEP_EP_VERSION}-cp311-cp311-linux_aarch64.whl" && \
    test -f /tmp/sgl-kernel-npu-wheels/torch_memory_saver-0.0.8-cp311-cp311-linux_aarch64.whl && \
    test -f "/tmp/sgl-kernel-npu-wheels/sgl_kernel_npu-${SGL_KERNEL_NPU_VERSION}-cp311-cp311-linux_aarch64.whl" && \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        "/tmp/sgl-kernel-npu-wheels/deep_ep-${DEEP_EP_VERSION}-cp311-cp311-linux_aarch64.whl" \
        /tmp/sgl-kernel-npu-wheels/torch_memory_saver-0.0.8-cp311-cp311-linux_aarch64.whl \
        "/tmp/sgl-kernel-npu-wheels/sgl_kernel_npu-${SGL_KERNEL_NPU_VERSION}-cp311-cp311-linux_aarch64.whl" && \
    deep_ep_site="$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && \
    test -n "${deep_ep_site}" && \
    cd "${deep_ep_site}" && \
    rm -f ./deep_ep_cpp*.so && \
    ln -sf deep_ep/deep_ep_cpp*.so && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c "import sys; import triton.language as tl; import triton.language.extra.cann.extension as al; ok=not hasattr(tl, 'insert_slice') and hasattr(al, 'insert_slice'); sys.exit('triton-ascend insert_slice API mismatch') if not ok else print('triton-ascend insert_slice API OK')" && \
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
RUN test -n "${SGLANG_FL_REVISION}" && \
    rm -rf /opt/sglang-plugin-fl && \
    mkdir -p /opt/sglang-plugin-fl && \
    printf '%s\n' "${SGLANG_FL_REVISION}" \
        > /opt/sglang-plugin-fl/.flagos-source-commit
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
      ai.flagos.sglang-plugin-fl.version="0.1.0" \
      ai.flagos.sglang-plugin-fl.revision="${SGLANG_FL_REVISION}" \
      ai.flagos.flaggems.revision="${FLAGGEMS_COMMIT}" \
      ai.flagos.flagcx.revision="${FLAGCX_COMMIT}" \
      ai.flagos.sgl-kernel-npu.release="${SGL_KERNEL_NPU_RELEASE}" \
      ai.flagos.sgl-kernel-npu.revision="${SGL_KERNEL_NPU_COMMIT}" \
      ai.flagos.sgl-kernel-npu.version="${SGL_KERNEL_NPU_VERSION}" \
      ai.flagos.sgl-kernel-npu.archive.sha256="${SGL_KERNEL_NPU_SHA256}" \
      ai.flagos.sgl-kernel-npu.solve-tril.sha256="${SGL_KERNEL_NPU_SOLVE_TRIL_SHA256}" \
      ai.flagos.deep-ep.version="${DEEP_EP_VERSION}" \
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
#     --build-arg SGLANG_FL_REVISION="$(git rev-parse HEAD)" \
#     --secret id=git_proxy,env=GIT_PROXY \
#     -f docker/ascend/empty-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.2.0-ascend-sglang0.5.18-ci .
# Push that tag first, then replace the CI reference with its registry digest.
