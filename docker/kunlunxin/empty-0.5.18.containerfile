# Kunlunxin stack install for SGLang 0.5.18 + sglang-plugin-FL + FlagTree +
# FlagGems + FlagCX. 
#
# - CUDA CI intentionally omits FlagCX (uses NCCL).
# - Kunlunxin empty images often KEEP FlagTree/Gems/CX from the base
#   and only rebuild SGLang + plugin.
# Adjust BASE_IMAGE, indexes, and commits for your platform before building.
#
# Build from the sglang-plugin-FL repository root:
#   docker build -f docs/Dockerfile.flagos-stack.example \
#     --build-arg SGLANG_FL_REVISION="$(git rev-parse HEAD)" \
#     -t sglang-fl:kunlunxin-0.5.18 .

ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-kunlunxin-tree_0.7.0rc3_xpu3.6-gems_5.3.0rc2-sglang_0.5.12-sglang_plugin_0.1.0-cx_0.13.0-python_3.10.12-torch_2.9.0-pcp_xrenone-gpu_p800-arc_amd64-driver_5.0.21.47:202610041703
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION=0.5.18
ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8

ARG SGLANG_FL_REVISION=dev/0.5.18

ARG FLAGTREE_VERSION=0.7.1a1+xpu3.6
ARG FLAGTREE_INDEX=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
ARG PYPI_INDEX=https://pypi.org/simple

ARG FLAGGEMS_COMMIT=75823ed20d21ff34f66cc54f5dd075e045913490
ARG FLAGGEMS_VERSION=0.0.0+g75823ed

ARG FLAGCX_COMMIT=9e42db36582330b6655db04c16896e1ae3da5c05
# Pinned by FlagCX ${FLAGCX_COMMIT} gitlinks (third-party/*).
ARG FLAGCX_GTEST_COMMIT=e90fe2485641bab0d6af4500192dc503384950d1
ARG FLAGCX_JSON_COMMIT=a0e9fb1e638cfbb5b8b556b7c51eaa81977bad48

ENV DEBIAN_FRONTEND=noninteractive \
    SGLANG_BUILD_RUST_EXTS=none \
    FLAGCX_PATH=/opt/FlagCX \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1) FlagTree (provides the `triton` module used by FlagGems / SGLang)
# ---------------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
    while python3 -m pip show triton >/dev/null 2>&1; do \
        python3 -m pip uninstall -y triton; \
    done && \
    python3 -m pip install --no-cache-dir --no-deps \
        --index-url "${FLAGTREE_INDEX}" \
        "flagtree===${FLAGTREE_VERSION}"

# ---------------------------------------------------------------------------
# 2) FlagGems (source editable; --no-deps keeps FlagTree / torch intact)
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        --index-url "${PYPI_INDEX}" \
        'setuptools>=68,<77' 'setuptools-scm>=8,<10' wheel \
        'packaging>=26' PyYAML sqlalchemy==2.0.48 && \
    rm -rf /opt/FlagGems && mkdir -p /opt/FlagGems && \
    curl -fL --retry 3 \
        "https://codeload.github.com/flagos-ai/FlagGems/tar.gz/${FLAGGEMS_COMMIT}" \
        -o /tmp/flaggems.tar.gz && \
    tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems && \
    printf '%s\n' "${FLAGGEMS_COMMIT}" > /opt/FlagGems/.flagos-source-commit && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=${FLAGGEMS_VERSION} \
        python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
            -e /opt/FlagGems && \
    rm -f /tmp/flaggems.tar.gz

# ---------------------------------------------------------------------------
# 3) SGLang 0.5.18 (empty / torch-free extra). Pin your vendor torch first.
#    Example constraint file — replace with the real vendor torch/musa/npu pins.
# ---------------------------------------------------------------------------
RUN rm -rf /opt/sglang-${SGLANG_VERSION} && mkdir -p /opt/sglang-${SGLANG_VERSION} && \
    curl -fL --retry 3 \
        "https://codeload.github.com/sgl-project/sglang/tar.gz/${SGLANG_COMMIT}" \
        -o /tmp/sglang.tar.gz && \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /opt/sglang-${SGLANG_VERSION} && \
    printf '%s\n' "${SGLANG_COMMIT}" > /opt/sglang-${SGLANG_VERSION}/.flagos-source-commit && \
    cd /opt/sglang-${SGLANG_VERSION}/python && \
    cp pyproject_other.toml pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
        python3 -m pip install --no-cache-dir \
            --index-url "${PYPI_INDEX}" \
            '.[srt_empty]' && \
    # python3 -m pip install --no-cache-dir --no-deps \
    #     xgrammar==0.2.1 compressed-tensors==0.15.0 && \
    rm -f /tmp/sglang.tar.gz

# ---------------------------------------------------------------------------
# 4) FlagCX via codeload tarballs (avoid git clone / submodule TLS flakes).
#    Backend flags are platform-specific:
#      Ascend:  make USE_ASCEND=1 CCL_HOME=$ASCEND_TOOLKIT_HOME
#      CUDA:    often omitted in FlagOS CUDA CI (NCCL instead)
#      Others:  follow vendor empty image / FlagCX docs
# ---------------------------------------------------------------------------
RUN rm -rf /opt/FlagCX && mkdir -p /opt/FlagCX && \
    curl -fL --retry 5 --retry-delay 5 \
        "https://codeload.github.com/flagos-ai/FlagCX/tar.gz/${FLAGCX_COMMIT}" \
        -o /tmp/flagcx.tar.gz && \
    tar -xzf /tmp/flagcx.tar.gz --strip-components=1 -C /opt/FlagCX && \
    printf '%s\n' "${FLAGCX_COMMIT}" > /opt/FlagCX/.flagos-source-commit && \
    rm -f /tmp/flagcx.tar.gz
RUN mkdir -p /opt/FlagCX/third-party/googletest /opt/FlagCX/third-party/json && \
    curl -fL --retry 5 --retry-delay 5 \
        "https://codeload.github.com/google/googletest/tar.gz/${FLAGCX_GTEST_COMMIT}" \
        -o /tmp/gtest.tar.gz && \
    tar -xzf /tmp/gtest.tar.gz --strip-components=1 -C /opt/FlagCX/third-party/googletest && \
    rm -f /tmp/gtest.tar.gz && \
    curl -fL --retry 5 --retry-delay 5 \
        "https://codeload.github.com/nlohmann/json/tar.gz/${FLAGCX_JSON_COMMIT}" \
        -o /tmp/json.tar.gz && \
    tar -xzf /tmp/json.tar.gz --strip-components=1 -C /opt/FlagCX/third-party/json && \
    rm -f /tmp/json.tar.gz
RUN python -m pip install "setuptools_scm>=8" wheel packaging && \
    cd /opt/FlagCX && \
    FLAGCX_ADAPTOR=klx \
    CCL_HOME=/opt/kunlun \
    DEVICE_HOME=/opt/kunlun \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAGCX="0.0.0+g9e42db3" \
    python -m pip install . -v --no-build-isolation --no-deps && \
    test -f /opt/FlagCX/build/lib/libflagcx.so

# ---------------------------------------------------------------------------
# 5) sglang-plugin-FL (codeload; pin SGLANG_FL_REVISION to a full commit SHA)
# ---------------------------------------------------------------------------
# Install sglang-plugin-FL from git (dev/0.5.18), then editable install.
ADD . /opt/sglang-plugin-FL
RUN cd /opt/sglang-plugin-FL && \
    python -m pip install --index-url "$PYPI_INDEX" -e . && \
    python .github/scripts/kunlunxin/verify_environment.py --check-versions


ENV USE_FLAGGEMS=1 \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_FL_DIST_BACKEND=flagcx 

WORKDIR /opt/sglang-plugin-FL