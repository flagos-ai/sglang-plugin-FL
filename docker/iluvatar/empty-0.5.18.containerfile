# Preserve the documented BI-V150 CoreX/torch/FlagCX runtime.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3-iluvatar-tree_0.6.2a3_iluvatar3.6-gems_0.0.0-sglang_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.12.11-torch_2.10.0_corex.4.5.0.20260629-pcp_ixml4.4.0-gpu_iluvatar001-arc_amd64-driver_4.5.0:202609231733
FROM ${BASE_IMAGE} AS runtime

ENV SGLANG_BUILD_RUST_EXTS=none \
    SGLANG_PLATFORM=sglang_fl \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=0 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/sgl-workspace/FlagCX \
    USE_FLAGGEMS=1 \
    TORCH_COMPILE_DISABLE=1

RUN if ! command -v python >/dev/null 2>&1; then \
        ln -s "$(command -v python3)" /usr/local/bin/python; \
    fi

# Pin existing vendor distributions before resolving pure-Python dependencies.
# A stock PyTorch or Triton wheel cannot replace the CoreX toolchain.
RUN python -c "from importlib.metadata import version; from pathlib import Path; \
    Path('/tmp/corex-constraints.txt').write_text(''.join( \
    f'{name}=={version(name)}\n' for name in ('torch', 'torchvision', 'torchaudio')))" && \
    python -m pip uninstall -y sglang sglang_fl flag_gems triton flagtree \
        sglang-kernel sgl-kernel flashinfer-python flash-attn ixformer && \
    python -m pip install --no-deps 'flagtree==0.6.2a3+iluvatar3.6' \
        --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

ARG FLAGGEMS_COMMIT=f7c55cb2fe1fb90fc713eafa6f63d7cbb73453a9
RUN mkdir -p /opt/FlagGems && \
    curl -fL --retry 3 "https://codeload.github.com/flagos-ai/FlagGems/tar.gz/${FLAGGEMS_COMMIT}" \
        -o /tmp/flaggems.tar.gz && \
    tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.3.4 \
        python -m pip install -c /tmp/corex-constraints.txt -e /opt/FlagGems

# v0.5.18 already contains srt_empty; do not reapply the v0.5.11 patch.
ARG SGLANG_VERSION=v0.5.18
RUN mkdir -p /opt/sglang-0.5.18 && \
    curl -fL --retry 3 "https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/${SGLANG_VERSION}" \
        -o /tmp/sglang.tar.gz && \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /opt/sglang-0.5.18 && \
    cd /opt/sglang-0.5.18/python && \
    cp pyproject_other.toml pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18 \
        python -m pip install -c /tmp/corex-constraints.txt '.[srt_empty]' && \
    python -m pip install --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0 && \
    python -m pip install -c /tmp/corex-constraints.txt 'packaging>=26.0' pyyaml

COPY . /opt/sglang-plugin-fl
# Explicitly install bounded import shims only into this CoreX empty image.
RUN python -m pip install --no-deps /opt/sglang-plugin-fl/addon/iluvatar-compat && \
    python -m pip install --no-deps /opt/sglang-plugin-fl && \
    python -c "from importlib.metadata import version; \
    assert version('sglang') == '0.5.18'; \
    assert 'corex' in version('torch'); \
    assert version('flagtree') == '0.6.2a3+iluvatar3.6'; \
    assert version('flag_gems') == '5.3.4'"

WORKDIR /opt/sglang-plugin-fl

FROM runtime AS test
RUN python -m pip install -c /tmp/corex-constraints.txt \
    pytest pytest-timeout pytest-json-report pytest-cov ruff
