# Upgrade the documented MUSA empty runtime without replacing its vendor stack.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-mthreads-empty-tree_none-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.10.12-torch_2.9.0-pcp_musa4.3.5-gpu_s5000-arc_amd64-driver_3.3.5_server:202608282011
FROM ${BASE_IMAGE} AS runtime

# The base image's system Python contains an older plugin. Install into the
# inference venv as a regular package so it takes precedence over that copy.
ENV PATH=/root/.virtualenvs/sglang-0.5.6/bin:${PATH}
ENV SGLANG_BUILD_RUST_EXTS=none
ENV TORCH_COMPILE_DISABLE=1
ENV FLAGCX_PATH=/sgl-workspace/FlagCX
ENV PYTHONPATH=/opt/FlagGems/src

ARG FLAGTREE_VERSION=0.6.2a3+mthreads3.6
ARG FLAGGEMS_COMMIT=01433e8304d78ef6c16ce76fe68e51e16c0b4d66
# The archive has no Git metadata; this is a local snapshot version, not a release.
ARG FLAGGEMS_SNAPSHOT_VERSION=5.4.0.dev20260908+g01433e830
RUN python -m pip uninstall -y triton flagtree && \
    python -m pip install --no-deps "flagtree==${FLAGTREE_VERSION}" \
        --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple && \
    mkdir -p /opt/FlagGems && \
    curl -fL --retry 3 "https://codeload.github.com/flagos-ai/FlagGems/tar.gz/${FLAGGEMS_COMMIT}" \
        -o /tmp/flaggems.tar.gz && \
    tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=${FLAGGEMS_SNAPSHOT_VERSION} \
        python -m pip install --no-deps -e /opt/FlagGems && \
    python -m pip install 'packaging>=26.0' PyYAML==6.0.1 sqlalchemy==2.0.48

ARG SGLANG_VERSION=v0.5.18
RUN mkdir -p /opt/sglang-0.5.18 && \
    curl -fL --retry 3 "https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/${SGLANG_VERSION}" \
        -o /tmp/sglang.tar.gz && \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /opt/sglang-0.5.18 && \
    cd /opt/sglang-0.5.18/python && \
    cp pyproject_other.toml pyproject.toml && \
    printf 'torch==2.9.0\ntorch_musa==2.9.0\n' > /tmp/musa-constraints.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION#v} \
        python -m pip install -c /tmp/musa-constraints.txt '.[srt_empty]' && \
    python -m pip install --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0

# These two optional runtime packages are imported by SGLang 0.5.18 during
# server startup even for unquantized models. --no-deps preserves vendor torch.
COPY . /opt/sglang-plugin-fl
RUN python -m pip install --no-deps /opt/sglang-plugin-fl

WORKDIR /opt/sglang-plugin-fl

# Build with --target ci for the MUSA Actions runner. Keep dependency setup out
# of individual jobs so pip cannot silently replace the tested vendor stack.
# Reuse only the real benchmark dataset from the previous MUSA CI image.
FROM harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:4ba3e396e929efa2773b9fc49b1662046501fc5348f561f3b076ee1a91465eb3 AS benchmark-cache
FROM runtime AS ci
RUN python -m pip install --no-cache-dir \
    'setuptools>=68,<82' wheel pytest pytest-cov pytest-timeout \
    pytest-json-report pytest-metadata pyyaml ruff
ENV MTHREADS_VISIBLE_DEVICES=all \
    SGLANG_MUSA_FP32_TP_ALLREDUCE=1 \
    SGLANG_FL_PER_OP="topk=reference;gemma_rms_norm=reference;fused_moe=vendor;chunk_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule_packed_decode=vendor" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
# SGLang's random benchmark samples ShareGPT before truncating/repeating tokens.
# Its pinned cache remains available when Actions changes HOME to /github/home.
COPY --from=benchmark-cache /opt/hf-cache /opt/hf-cache
RUN python -c "from importlib.metadata import version; \
    assert version('sglang') == '0.5.18', version('sglang'); \
    assert version('torch') == '2.9.0', version('torch'); \
    assert version('flagtree') == '0.6.2a3+mthreads3.6', version('flagtree')"
WORKDIR /workspace
