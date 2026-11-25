# MetaX C550: upgrade the MetaX empty image to SGLang 0.5.18 empty mode without
# replacing its vendor stack (torch 2.10.0+metax3.9.0.8, FlagTree
# 0.7.0.post2+metax3.6, FlagGems 5.3.0rc2, FlagCX 0.13.0 at /sgl-workspace/FlagCX).
#
# Build from the repository root:
#   docker build -f docker/metax/empty-0.5.18.containerfile --target runtime -t <runtime-tag> .
#   docker build -f docker/metax/empty-0.5.18.containerfile --target ci      -t <ci-tag> .
# Mirrors can be passed with --build-arg BASE_IMAGE=... and PIP_INDEX_URL=....
# The build downloads the SGLang source from codeload.github.com (sha256-checked).
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-metax-empty-tree_0.7.0.post2_metax3.6-gems_5.3.0rc2-sglang_0.5.11-sglang_plugin_0.1.0-cx_0.13.0-python_3.12.11-torch_2.10.0_metax3.9.0.8-pcp_maca3.9.0.29-driver:202610311308
FROM ${BASE_IMAGE} AS runtime

# The base image activates /opt/conda only in interactive shells.
ENV PATH=/opt/conda/bin:${PATH}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG SGLANG_VERSION=0.5.18
ARG SGLANG_SHA256=4ed28e4a5bb0bee3a2fe905cea6a4db95299a9c289275bd7f75e617ff7cad361

# The base image carries SGLang 0.5.11 and an older plugin as editable checkouts
# under /sgl-workspace; the install script uninstalls them, and the stale
# sources are removed here so the new ones take the same paths.
COPY . /tmp/sglang-plugin-fl-src
RUN rm -rf /sgl-workspace/sglang /sgl-workspace/sglang-plugin-FL && \
    mv /tmp/sglang-plugin-fl-src /sgl-workspace/sglang-plugin-FL && \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHON=/opt/conda/bin/python \
    PLUGIN_DIR=/sgl-workspace/sglang-plugin-FL \
    SGLANG_DIR=/sgl-workspace/sglang \
    SGLANG_VERSION=${SGLANG_VERSION} \
    SGLANG_SHA256=${SGLANG_SHA256} \
    EDITABLE=1 \
        bash /sgl-workspace/sglang-plugin-FL/docker/metax/install_empty_0.5.18.sh && \
    rm -rf /root/.cache/pip

# Runtime defaults used by the validated MetaX empty-mode runs. The FlagGems
# blacklist is not set here: sglang_fl/dispatch/config/metax.yaml owns it, and
# SGLANG_FL_FLAGOS_BLACKLIST would replace (not extend) that list.
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true \
    FLAGCX_PATH=/sgl-workspace/FlagCX \
    SGLANG_FL_DIST_BACKEND=flagcx \
    MACA_SMALL_PAGESIZE_ENABLE=1 \
    MACA_QUEUE_SCHEDULE_POLICY=1 \
    MCCL_GRAPH_MIXING_SUPPORT=1 \
    PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1 \
    TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1 \
    TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

WORKDIR /sgl-workspace/sglang-plugin-FL

# Build with --target ci for the MetaX Actions runner. Keep dependency setup out
# of individual jobs so pip cannot silently replace the tested vendor stack;
# .github/scripts/metax/setup.sh only re-links the checked-out plugin.
FROM runtime AS ci
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url ${PIP_INDEX_URL} \
        pytest pytest-cov pytest-timeout pytest-json-report pytest-metadata pyyaml ruff && \
    python -c "from importlib.metadata import version as v; \
assert v('sglang') == '0.5.18', v('sglang'); \
assert v('torch') == '2.10.0+metax3.9.0.8', v('torch'); \
assert v('flagtree') == '0.7.0.post2+metax3.6', v('flagtree'); \
assert v('numpy') == '1.26.4', v('numpy')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /workspace
