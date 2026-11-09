# Build from the repository root. Stable dependencies precede all plugin sources.
# --target dependencies builds an image for iterating on the plugin without
# reinstalling the vendor stack or SGLang. Export/import BuildKit cache in CI.
FROM harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:ee5b2b9420dedc8eb6ad6658397e9d9602533f1c6e74a53b924ff976a113d4ab AS runtime

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/opt/accl-p:${LD_LIBRARY_PATH}

# FlagCX can be built without a device. Keep its source tree for the interservice
# Python wrapper consumed by the plugin; do not copy a development image rootfs.
FROM runtime AS flagcx
ARG FLAGCX_COMMIT=0beba7aa7e76cc0885a9929f1975857bcd310d5e
ARG BUILD_JOBS=4
RUN git init /opt/FlagCX && \
    git -C /opt/FlagCX remote add origin https://github.com/flagos-ai/FlagCX.git && \
    git -C /opt/FlagCX fetch --depth 1 origin "${FLAGCX_COMMIT}" && \
    git -C /opt/FlagCX checkout --detach FETCH_HEAD && \
    git -C /opt/FlagCX submodule update --init --depth 1 third-party/json
RUN make -C /opt/FlagCX -j"${BUILD_JOBS}" USE_DU=1 \
        DEVICE_HOME=/usr/local/PPU_SDK/CUDA_SDK \
        DEVICE_INCLUDE='/usr/local/PPU_SDK/CUDA_SDK/include /usr/local/PPU_SDK/CUDA_SDK/include/cccl' \
        CCL_HOME=/usr/local/PPU_SDK/CUDA_SDK && \
    test -f /opt/FlagCX/build/lib/libflagcx.so

FROM flagcx AS gems
# Match the development image's FlagGems source, while preserving runtime Torch
# and FlagTree. --no-deps prevents a public CUDA/Triton wheel replacing them.
ARG CI_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG FLAGGEMS_COMMIT=7536a18c00adbc29689a5657f2f5866fd637b23a
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    git init /opt/FlagGems && \
    git -C /opt/FlagGems remote add origin https://github.com/flagos-ai/FlagGems.git && \
    git -C /opt/FlagGems fetch --depth 1 origin "${FLAGGEMS_COMMIT}" && \
    git -C /opt/FlagGems checkout --detach FETCH_HEAD
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    PIP_INDEX_URL="${CI_PIP_INDEX_URL}" \
        SETUPTOOLS_SCM_PRETEND_VERSION=5.5.0.dev0+g7536a18c0 \
        python -m pip install --no-deps /opt/FlagGems

FROM gems AS vendor-kernel
ARG PPU_SGL_KERNEL_URL=https://mirrors.aliyun.com/pypi/packages/5a/c8/262263edac73965f190bffd4c072ec65705d9176486b10eda4d5632db1ab/sglang_kernel-0.4.2.post2-cp310-abi3-manylinux2014_x86_64.whl
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    python -m pip install --no-deps \
        "${PPU_SGL_KERNEL_URL}#sha256=f275ce01010933c4ea655e832a05442b766ab8f23dcfd6ff678b15e295d42203"

# The operations runtime intentionally omits torchvision, while SGLang imports
# torchvision.io.decode_jpeg at module load time.  The public 0.25.0 wheel is
# not ABI-compatible with the vendor PPU Torch (its C++ operators do not load),
# so build the exact version used by the development image against runtime
# Torch.  Keep compilers and headers in a throw-away stage; only the wheel is
# copied into the final image.  CUDA image operators are not needed by SGLang's
# text CI and cannot be compiled on a device-less GitHub builder.
FROM vendor-kernel AS torchvision-builder
ARG CI_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHVISION_COMMIT=8ac84ee75afb1c327902156b5336f56ad63b7e2f
ARG BUILD_JOBS=4
RUN apt-get update && \
    apt-get install -y --no-install-recommends libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*
RUN git init /opt/vision && \
    git -C /opt/vision remote add origin https://github.com/pytorch/vision.git && \
    git -C /opt/vision fetch --depth 1 origin "${TORCHVISION_COMMIT}" && \
    git -C /opt/vision checkout --detach FETCH_HEAD
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    PIP_INDEX_URL="${CI_PIP_INDEX_URL}" \
        python -m pip install setuptools==69.5.1 wheel && \
    BUILD_VERSION="${TORCHVISION_VERSION}" MAX_JOBS="${BUILD_JOBS}" \
    TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_WEBP=0 \
        python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/vision-wheels /opt/vision

FROM vendor-kernel AS vision
COPY --from=torchvision-builder /opt/vision-wheels/torchvision-*.whl /tmp/
RUN python -m pip install --no-deps /tmp/torchvision-*.whl && \
    rm -f /tmp/torchvision-*.whl && \
    python -c "import torchvision; from torchvision.io import decode_jpeg; assert torchvision.__version__.split('+')[0] == '0.25.0'"

# Snapshot the vendor constraints once, before adding generic Python packages.
RUN python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
packages = ('torch', 'torchvision', 'torchaudio', 'torchcodec', 'flagtree',
            'flag_gems', 'sglang-kernel', 'cuda-python', 'cuda-bindings',
            'apache-tvm-ffi', 'torch_memory_saver', 'numpy')
lines = ['triton<0']
for name in packages:
    try:
        lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
Path('/opt/vendor-constraints.txt').write_text('\n'.join(lines) + '\n')
PY

FROM vision AS sglang
ARG SGLANG_VERSION=0.5.18
ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
RUN git init /opt/sglang && \
    git -C /opt/sglang remote add origin https://github.com/sgl-project/sglang.git && \
    git -C /opt/sglang fetch --depth 1 origin "${SGLANG_COMMIT}" && \
    git -C /opt/sglang checkout --detach FETCH_HEAD && \
    cp /opt/sglang/python/pyproject_other.toml /opt/sglang/python/pyproject.toml
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    PIP_INDEX_URL="${CI_PIP_INDEX_URL}" SGLANG_BUILD_RUST_EXTS=none \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
    python -m pip install --no-build-isolation -c /opt/vendor-constraints.txt \
        -e '/opt/sglang/python[srt_empty]' && \
    PIP_INDEX_URL="${CI_PIP_INDEX_URL}" \
        python -m pip install --no-deps xgrammar==0.2.1

FROM sglang AS dependencies
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    PIP_INDEX_URL="${CI_PIP_INDEX_URL}" \
        python -m pip install -c /opt/vendor-constraints.txt \
        pytest pytest-cov pytest-timeout==2.4.0 pytest-json-report \
        pytest-metadata requests ruff pyyaml && \
    python -m pip freeze > /opt/ci-requirements.freeze.txt

# Frequently changed plugin code and CI defaults are deliberately last. Tests,
# docs, local reports, Git metadata, and unrelated platforms are not build inputs.
FROM dependencies AS ci
ENV PYTHONPATH=/opt/sglang/python:/opt/flagtree \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/opt/FlagCX \
    SGLANG_IS_FLASHINFER_AVAILABLE=false \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /opt/sglang-plugin-fl
COPY pyproject.toml ./
COPY sglang_fl/ ./sglang_fl/
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation -e . && \
    python -c "from importlib.metadata import version; assert version('sglang') == '0.5.18'; assert version('flagtree') == '0.7.0+ppu3.6'" && \
    test -f /opt/FlagCX/plugin/interservice/flagcx_wrapper.py
