# Upgrade the documented MUSA empty runtime without replacing its vendor stack.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-mthreads-empty-tree_none-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.10.12-torch_2.9.0-pcp_musa4.3.5-gpu_s5000-arc_amd64-driver_3.3.5_server:202608282011
FROM ${BASE_IMAGE}

# The base image's system Python contains an older plugin. Install into the
# inference venv as a regular package so it takes precedence over that copy.
ENV PATH=/root/.virtualenvs/sglang-0.5.6/bin:${PATH}
ENV SGLANG_BUILD_RUST_EXTS=none
ENV TORCH_COMPILE_DISABLE=1
ENV FLAGCX_PATH=/sgl-workspace/FlagCX

ARG SGLANG_VERSION=v0.5.18
RUN git clone --depth 1 --branch ${SGLANG_VERSION} \
        https://github.com/sgl-project/sglang.git /opt/sglang-0.5.18 && \
    cd /opt/sglang-0.5.18/python && \
    cp pyproject_other.toml pyproject.toml && \
    printf 'torch==2.9.0\ntorch_musa==2.9.0\ntriton==3.2.0\n' > /tmp/musa-constraints.txt && \
    python -m pip install -c /tmp/musa-constraints.txt '.[srt_empty]' && \
    python -m pip install --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0

# These two optional runtime packages are imported by SGLang 0.5.18 during
# server startup even for unquantized models. --no-deps preserves vendor torch.
COPY . /opt/sglang-plugin-fl
RUN python -m pip install --no-deps /opt/sglang-plugin-fl

WORKDIR /opt/sglang-plugin-fl
