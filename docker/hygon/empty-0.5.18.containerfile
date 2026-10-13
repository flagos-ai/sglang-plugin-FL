# Upgrade the documented Hygon empty image while preserving its DTK wheels.
# Build from the repository root:
# docker build -f docker/hygon/empty-0.5.18.containerfile -t sglang-fl:hygon-0.5.18 .
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-hygon-empty-tree_0.6.1_hcu3.6-gems_5.4.0.dev0-sglang_0.5.11_empty-plugin_0.1.0-cx_0.13.0-python_3.10.12-torch_2.10.0_das.opt1.dtk2604.2606151054.g98b1a8-pcp_dtk-non:202609151101
FROM ${BASE_IMAGE} AS runtime

ENV SGLANG_BUILD_RUST_EXTS=none \
    FLAGCX_PATH=/sgl-workspace/FlagCX

COPY .github/scripts/hygon/verify_environment.py /opt/hygon-verify-environment.py
# Do not guess PEP 440 local versions from image tags: capture installed metadata.
RUN python /opt/hygon-verify-environment.py \
        --capture-vendor-stack /opt/hygon-vendor-baseline.json

ARG SGLANG_VERSION=0.5.18
RUN mkdir -p /opt/sglang-0.5.18 && \
    curl -fL --retry 3 "https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/v${SGLANG_VERSION}" \
        -o /tmp/sglang.tar.gz && \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /opt/sglang-0.5.18 && \
    cd /opt/sglang-0.5.18/python && \
    cp pyproject_other.toml pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
        python -m pip install --no-build-isolation \
        -c /opt/hygon-vendor-baseline.constraints.txt '.[srt_empty]' \
        xgrammar==0.2.1 compressed-tensors==0.15.0 sqlalchemy==2.0.48

# srt_empty uses the hardware-independent runtime_base dependency subset.
# The captured constraints preserve DTK torch and the vendor kernel stack;
# dependency conflicts fail instead of replacing those builds.
# xgrammar/compressed-tensors/SQLAlchemy are also imported during startup.
# Install only the checked-out plugin after resolving Python dependencies.
COPY . /opt/sglang-plugin-fl
RUN python -m pip install --no-deps --no-build-isolation -e /opt/sglang-plugin-fl && \
    python /opt/sglang-plugin-fl/.github/scripts/hygon/verify_environment.py \
        --plugin-root /opt/sglang-plugin-fl \
        --baseline /opt/hygon-vendor-baseline.json --metadata-only

ENV USE_FLAGGEMS=1 \
    USE_FLAGTUNE=0 \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_FL_DIST_BACKEND=flagcx

WORKDIR /opt/sglang-plugin-fl
# A build passes metadata checks only. Run the full verifier with --require-gpu
# on BW1000 before model acceptance. A source/version lock does not certify
# import compatibility, vendor kernel execution, or TP/PP communication.
