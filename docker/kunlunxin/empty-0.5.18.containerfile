# Rebuild SGLang 0.5.18 + sglang-plugin-FL on the FlagOS Kunlunxin empty stack.
# Preserves vendor Torch-2.9.0、FlagTree-0.7.1a1+xpu3.6、FlagGems-0.0+g7fcdbed07、FlagCX-master from the base image.
#
# Build from the repository root (sglang-plugin-FL):
#   docker build -f docker/kunlunxin/empty-0.5.18.containerfile \
#     -t sglang-fl:kunlunxin-0.5.18 .
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-kunlunxin-tree_0.7.1a1_xpu3.6-gems_master_g7fcdbed07-sglang_0.5.18-sglang-plugin_dev_0.5.18-cx_master_g7812d2e9-python_3.10.12-torch_2.9.0-pcp_xrenone-gpu_p800-arc_amd64-driver_5.0.21.47:202610271659
FROM ${BASE_IMAGE} AS runtime

# Skip Rust extensions in the image build (cargo/proxy often unavailable).
# Text inference / TP / serve smoke do not require _multimodal.
ENV USE_FLAGGEMS=1 \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/sgl-workspace/sglang-upgrade/FlagCX

# Optional trusted mirror/cache for sites with slow access to GitHub.
ARG SGLANG_VERSION=0.5.18
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch /opt/sglang-v0.5.18-kunlunxin.patch

# Install SGLang v0.5.18 from source: download tag tarball, apply Kunlunxin patch,
# then pip install .[srt_empty].    
RUN rm -rf /sgl-workspace/sglang-upgrade/sglang && \
  git clone --branch "v${SGLANG_VERSION}" --single-branch \
    https://github.com/sgl-project/sglang.git \
    /sgl-workspace/sglang-upgrade/sglang && \
  cd /sgl-workspace/sglang-upgrade/sglang && \
  git config --global --add safe.directory /sgl-workspace/sglang-upgrade/sglang && \
  git apply --check /opt/sglang-v0.5.18-kunlunxin.patch && \
  git apply /opt/sglang-v0.5.18-kunlunxin.patch && \
  (python -m pip uninstall -y sglang sglang_fl || true) && \
  cd python && \
  python -m pip install \
    --index-url "$PIP_INDEX_URL" '.[srt_empty]'

# Install sglang-plugin-FL from git (dev/0.5.18), then editable install.
ADD . /sgl-workspace/sglang-upgrade/sglang-plugin-FL
RUN cd /sgl-workspace/sglang-upgrade/sglang-plugin-FL && \
    python -m pip install --index-url "$PIP_INDEX_URL" -e . && \
    python .github/scripts/kunlunxin/verify_environment.py --check-versions

WORKDIR /tmp
